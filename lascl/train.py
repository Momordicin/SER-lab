import os
import json
from typing import Dict
import random
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from .config import LaSCLConfig
from .dataset import (
    get_label_names_from_train,
    load_split_items,
    compute_class_weights,
    LaSCLDataset,
    LaSCLCollator,
)
from .model import LaSCLModel
from .losses import LaSCLLoss


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
def evaluate(model, criterion, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_scl = 0.0
    total_items = 0

    y_true = []
    y_pred = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        outputs = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            aug_input_values=batch["aug_input_values"],
            aug_attention_mask=batch["aug_attention_mask"],
            labels=labels,
        )

        loss_dict = criterion(outputs, labels)
        bs = labels.size(0)

        total_loss += loss_dict["loss"].item() * bs
        total_ce += loss_dict["ce_loss"].item() * bs
        total_scl += loss_dict["scl_loss"].item() * bs
        total_items += bs

        preds = torch.argmax(outputs["logits"], dim=-1)
        y_true.append(labels.cpu())
        y_pred.append(preds.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    acc = (y_true == y_pred).float().mean().item()

    # macro F1 without sklearn dependency here would be annoying,
    # so keep val metric simple for now
    return {
        "loss": total_loss / max(total_items, 1),
        "ce_loss": total_ce / max(total_items, 1),
        "scl_loss": total_scl / max(total_items, 1),
        "accuracy": acc,
    }


def stratified_take(items, max_items, seed=42):
    """
    Take up to max_items from (path, label_id) items while preserving
    class diversity as much as possible.

    Args:
        items: list of (path, label_id)
        max_items: target number of items to keep
        seed: random seed for deterministic selection
    """
    if max_items is None or max_items >= len(items):
        return items

    by_label = defaultdict(list)
    for item in items:
        _, y = item
        by_label[y].append(item)

    rng = random.Random(seed)
    for y in by_label:
        rng.shuffle(by_label[y])

    num_labels = len(by_label)
    per_class = max(1, max_items // num_labels)

    selected = []
    for y in sorted(by_label):
        selected.extend(by_label[y][:per_class])

    # If we still need more items, fill from leftovers
    if len(selected) < max_items:
        leftovers = []
        for y in sorted(by_label):
            leftovers.extend(by_label[y][per_class:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_items - len(selected)])

    rng.shuffle(selected)
    return selected[:max_items]


def train_lascl(
    data_root: str,
    out_dir: str,
    cfg: LaSCLConfig,
    max_train_items: int | None = None,
    max_val_items: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    labels = get_label_names_from_train(train_dir)
    print("Labels found in train split:", labels)

    if labels != cfg.label_texts:
        print("Warning: labels in folder do not exactly match cfg.label_texts")
        print("  train labels:", labels)
        print("  cfg labels  :", cfg.label_texts)

    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    processor = AutoFeatureExtractor.from_pretrained(cfg.audio_model_name)

    train_items = load_split_items(train_dir, label2id)
    val_items = load_split_items(val_dir, label2id)

    train_items = stratified_take(train_items, max_train_items, seed=cfg.seed)
    val_items = stratified_take(val_items, max_val_items, seed=cfg.seed)

    print(f"Using {len(train_items)} train items and {len(val_items)} val items")
    print("Train subset label counts:", Counter([y for _, y in train_items]))
    print("Val subset label counts:", Counter([y for _, y in val_items]))

    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_items, num_labels=len(labels))

    train_ds = LaSCLDataset(
        train_items,
        target_sr=cfg.sampling_rate,
        max_seconds=cfg.max_seconds,
        min_seconds=cfg.min_seconds,
        use_augmentation=True,
    )
    val_ds = LaSCLDataset(
        val_items,
        target_sr=cfg.sampling_rate,
        max_seconds=cfg.max_seconds,
        min_seconds=cfg.min_seconds,
        use_augmentation=False,
    )

    collator = LaSCLCollator(processor=processor, sampling_rate=cfg.sampling_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")

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

    criterion = LaSCLLoss(
        num_labels=len(labels),
        temperature=cfg.temperature,
        lambda_ce=cfg.lambda_ce,
        lambda_scl=cfg.lambda_scl,
        class_weights=class_weights.to(device) if class_weights is not None else None,
    )

    # separate parameter groups
    audio_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("audio_encoder"):
            audio_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": audio_params, "lr": cfg.audio_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")

    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_scl = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            labels_t = batch["labels"]

            optimizer.zero_grad()

            outputs = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
                aug_input_values=batch["aug_input_values"],
                aug_attention_mask=batch["aug_attention_mask"],
                labels=labels_t,
            )

            loss_dict = criterion(outputs, labels_t)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            bs = labels_t.size(0)
            running_loss += loss.item() * bs
            running_ce += loss_dict["ce_loss"].item() * bs
            running_scl += loss_dict["scl_loss"].item() * bs
            seen += bs

            if step % 20 == 0:
                print(
                    f"Epoch {epoch} Step {step} | "
                    f"loss={running_loss/seen:.4f} "
                    f"ce={running_ce/seen:.4f} "
                    f"scl={running_scl/seen:.4f}"
                )

        train_metrics = {
            "loss": running_loss / max(seen, 1),
            "ce_loss": running_ce / max(seen, 1),
            "scl_loss": running_scl / max(seen, 1),
        }
        val_metrics = evaluate(model, criterion, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        print(f"\nEpoch {epoch} complete")
        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "label2id": label2id,
                    "id2label": id2label,
                },
                best_path,
            )
            processor.save_pretrained(out_dir)
            print(f"Saved best model to {best_path}")

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
