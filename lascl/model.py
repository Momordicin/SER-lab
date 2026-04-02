import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoTokenizer,
    WavLMModel,
)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class LaSCLModel(nn.Module):
    def __init__(
        self,
        audio_model_name: str = "microsoft/wavlm-base-plus",
        text_model_name: str = "roberta-base",
        label_texts=None,
        proj_dim: int = 256,
        proj_hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()

        if label_texts is None:
            label_texts = ["angry", "happy", "neutral", "sad"]

        self.label_texts = label_texts
        self.num_labels = len(label_texts)

        # Audio encoder
        self.audio_encoder = WavLMModel.from_pretrained(audio_model_name)
        audio_hidden = self.audio_encoder.config.hidden_size

        if freeze_feature_encoder:
            for p in self.audio_encoder.feature_extractor.parameters():
                p.requires_grad = False

        # Text encoder
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text_encoder.config.hidden_size

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Projection heads
        self.audio_proj = ProjectionHead(
            in_dim=audio_hidden,
            out_dim=proj_dim,
            hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )
        self.text_proj = ProjectionHead(
            in_dim=text_hidden,
            out_dim=proj_dim,
            hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )

        # Classification head for CE loss
        self.classifier = nn.Linear(audio_hidden, self.num_labels)

    def mean_pool(self, hidden_states, attention_mask=None):
        # hidden_states: [B, T_feat, D]
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T_feat, 1]
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_audio(self, input_values, attention_mask=None):
        out = self.audio_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = out.last_hidden_state  # [B, T_feat, D]

        feat_attention_mask = None
        if attention_mask is not None:
            feat_attention_mask = self.audio_encoder._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )

        pooled = self.mean_pool(hidden_states, feat_attention_mask)
        return pooled

    def encode_text_labels(self, device):
        toks = self.text_tokenizer(
            self.label_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        out = self.text_encoder(**toks)
        # CLS-style first token pooling
        pooled = out.last_hidden_state[:, 0, :]
        return pooled

    def forward(
        self,
        input_values,
        attention_mask=None,
        aug_input_values=None,
        aug_attention_mask=None,
        labels=None,
    ):
        device = input_values.device

        # Original audio branch
        audio_hidden = self.encode_audio(input_values, attention_mask)          # [B, Da]
        audio_z = self.audio_proj(audio_hidden)                                 # [B, d]
        logits = self.classifier(audio_hidden)                                  # [B, K]

        # Augmented audio branch
        aug_audio_hidden = None
        aug_audio_z = None
        if aug_input_values is not None:
            aug_audio_hidden = self.encode_audio(aug_input_values, aug_attention_mask)
            aug_audio_z = self.audio_proj(aug_audio_hidden)

        # Text label branch
        text_hidden = self.encode_text_labels(device)                           # [K, Dt]
        text_z = self.text_proj(text_hidden)                                    # [K, d]

        return {
            "audio_hidden": audio_hidden,
            "audio_z": audio_z,
            "logits": logits,
            "aug_audio_hidden": aug_audio_hidden,
            "aug_audio_z": aug_audio_z,
            "text_hidden": text_hidden,
            "text_z": text_z,
            "labels": labels,
        }
