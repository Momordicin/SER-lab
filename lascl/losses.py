import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """
    Standard supervised contrastive loss.

    Args:
        features: [M, d] normalized embeddings
        labels:   [M] integer labels
    """
    device = features.device
    labels = labels.view(-1)
    assert features.ndim == 2
    assert labels.ndim == 1
    assert features.size(0) == labels.size(0)

    # cosine similarity because features are L2-normalized
    logits = torch.matmul(features, features.T) / temperature   # [M, M]

    # numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).to(device)   # [M, M]
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=device)

    # positives exclude self
    positive_mask = mask & (~self_mask)
    negative_mask = ~self_mask

    exp_logits = torch.exp(logits) * negative_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0

    mean_log_prob_pos = torch.zeros(features.size(0), device=device, dtype=features.dtype)
    mean_log_prob_pos[valid] = (
        (positive_mask[valid] * log_prob[valid]).sum(dim=1) /
        positive_counts[valid].clamp(min=1)
    )

    loss = -mean_log_prob_pos[valid].mean()
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=device, dtype=features.dtype)
    return loss


class LaSCLLoss(nn.Module):
    """
    First runnable LaSCL-lite loss:
      total = lambda_ce * CE + lambda_scl * SCL

    SCL is computed over:
      - original audio embeddings
      - augmented audio embeddings
      - text label embeddings
    """
    def __init__(
        self,
        num_labels: int,
        temperature: float = 0.07,
        lambda_ce: float = 0.5,
        lambda_scl: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.temperature = temperature
        self.lambda_ce = lambda_ce
        self.lambda_scl = lambda_scl
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, outputs: dict, labels: torch.Tensor):
        logits = outputs["logits"]              # [B, K]
        audio_z = outputs["audio_z"]            # [B, d]
        aug_audio_z = outputs["aug_audio_z"]    # [B, d]
        text_z = outputs["text_z"]              # [K, d]

        device = logits.device
        labels = labels.to(device)

        # CE loss on original audio branch
        ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        ce_loss = ce_loss_fn(logits, labels)

        # Build contrastive batch:
        #   original audio + augmented audio + text label embeddings
        # labels for text embeddings are [0,1,...,K-1]
        text_labels = torch.arange(self.num_labels, device=device, dtype=labels.dtype)

        contrast_features = torch.cat([audio_z, aug_audio_z, text_z], dim=0)           # [2B + K, d]
        contrast_labels = torch.cat([labels, labels, text_labels], dim=0)               # [2B + K]

        scl_loss = supervised_contrastive_loss(
            contrast_features,
            contrast_labels,
            temperature=self.temperature,
        )

        total_loss = self.lambda_ce * ce_loss + self.lambda_scl * scl_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "scl_loss": scl_loss.detach(),
        }