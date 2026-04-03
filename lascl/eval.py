import os
import json
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .config import LaSCLConfig
from .dataset import (
    get_label_names_from_train,
    load_split_items,
    LaSCLDataset,
    LaSCLCollator,
)
from .model import LaSCLModel


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_lascl(
    data_root: str,
    ckpt_path: str,
    cfg: LaSCLConfig,
    split: str = "test",
    mode: str = "nearest",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if mode not in {"nearest", "classifier"}:
        raise ValueError("mode must be 'nearest' or 'classifier'")

    train_dir = os.path.join(data_root, "train")
    split_dir = os.path.join(data_root, split)

    labels = get_label_names_from_train(train_dir)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    processor = AutoFeatureExtractor.from_pretrained(cfg.audio_model_name)

    items = load_split_items(split_dir, label2id)
    ds = LaSCLDataset(
        items,
        target_sr=cfg.sampling_rate,
        max_seconds=cfg.max_seconds,
        min_seconds=cfg.min_seconds,
        use_augmentation=False,
    )
    collator = LaSCLCollator(processor=processor, sampling_rate=cfg.sampling_rate)

    dl = DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    model = LaSCLModel(
        audio_model_name=cfg.audio_model_name,
        text_model_name=cfg.text_model_name,
        label_texts=cfg.label_texts,
        proj_dim=cfg.proj_dim,
        proj_hidden_dim=cfg.proj_hidden_dim,
        dropout=cfg.dropout,
        freeze_feature_encoder=cfg.freeze_feature_encoder,
        freeze_text_encoder=cfg.freeze_text_encoder,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Precompute label embeddings once
    text_z = None
    if mode == "nearest":
        text_hidden = model.encode_text_labels(device)
        text_z = model.text_proj(text_hidden)   # [K, d]

    y_true = []
    y_pred = []

    for batch in dl:
        batch = move_batch_to_device(batch, device)
        labels_t = batch["labels"]

        audio_hidden = model.encode_audio(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
        )

        if mode == "nearest":
            audio_z = model.audio_proj(audio_hidden)   # [B, d]
            sims = torch.matmul(audio_z, text_z.T)     # [B, K]
            preds = torch.argmax(sims, dim=-1)
        else:
            logits = model.classifier(audio_hidden)    # [B, K]
            preds = torch.argmax(logits, dim=-1)

        y_true.append(labels_t.cpu())
        y_pred.append(preds.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")

    print(f"\nLaSCL evaluation on split='{split}', mode='{mode}'")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 macro     : {f1m:.4f}")
    print(f"F1 weighted  : {f1w:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "labels": labels,
    }