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


def label_divergence_loss(label_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Encourage projected label embeddings to be separated from each other.

    Args:
        label_embeddings: [K, d], assumed normalized

    Returns:
        scalar loss
    """
    device = label_embeddings.device
    K = label_embeddings.size(0)

    if K < 2:
        return torch.tensor(0.0, device=device, dtype=label_embeddings.dtype)

    sim = torch.matmul(label_embeddings, label_embeddings.T)   # [K, K]

    losses = []
    for i in range(K):
        others_mask = torch.ones(K, dtype=torch.bool, device=device)
        others_mask[i] = False

        sims_i = sim[i, others_mask]   # [K-1]
        denom = torch.exp(sims_i).sum() + 1.0
        p = torch.exp(sims_i) / denom

        loss_i = -torch.log((1.0 - p).clamp(min=1e-12))
        losses.append(loss_i.mean())

    return torch.stack(losses).mean()


class LaSCLLoss(nn.Module):
    """
    LaSCL training loss:
      total = lambda_ce * CE
            + lambda_scl * SCL
            + lambda_label_div * LabelDivergence

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
        lambda_label_div: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.temperature = temperature
        self.lambda_ce = lambda_ce
        self.lambda_scl = lambda_scl
        self.lambda_label_div = lambda_label_div
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

        label_div_loss = label_divergence_loss(text_z)

        total_loss = (
            self.lambda_ce * ce_loss
            + self.lambda_scl * scl_loss
            + self.lambda_label_div * label_div_loss
        )

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "scl_loss": scl_loss.detach(),
            "label_div_loss": label_div_loss.detach(),
        }