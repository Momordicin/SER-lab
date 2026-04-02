import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor

from .augment import augment_waveform


EMO_TAGS = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "EXC": "happy",   # if needed later
}
EMO_RE = re.compile(r"_(ANG|HAP|NEU|SAD|EXC)(?:_|$)")


def is_real_wav(path: str) -> bool:
    base = os.path.basename(path)
    return base.lower().endswith(".wav") and not base.startswith("._")


def list_wavs_recursive(root: str) -> List[str]:
    return [
        p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        if is_real_wav(p)
    ]


def extract_label_from_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = EMO_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse emotion from filename: {os.path.basename(path)}")
    tag = m.group(1)
    return EMO_TAGS[tag]


def get_label_names_from_train(train_dir: str) -> List[str]:
    labels = []
    for name in sorted(os.listdir(train_dir)):
        p = os.path.join(train_dir, name)
        if os.path.isdir(p) and list_wavs_recursive(p):
            labels.append(name.lower())
    if not labels:
        raise RuntimeError(f"No label folders found in {train_dir}")
    return labels


def load_split_items(split_dir: str, label2id: Dict[str, int]) -> List[Tuple[str, int]]:
    items = []
    for lab in sorted(os.listdir(split_dir)):
        lab_dir = os.path.join(split_dir, lab)
        if not os.path.isdir(lab_dir):
            continue
        lab_norm = lab.lower()
        if lab_norm not in label2id:
            continue
        for p in list_wavs_recursive(lab_dir):
            items.append((p, label2id[lab_norm]))
    if not items:
        raise RuntimeError(f"No wav files found in {split_dir}")
    return items


def compute_class_weights(items: List[Tuple[str, int]], num_labels: int) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.int64)
    for _, y in items:
        counts[y] += 1
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights * (num_labels / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


class LaSCLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        target_sr: int = 16000,
        max_seconds: float = 6.0,
        min_seconds: float = 0.2,
        use_augmentation: bool = True,
    ):
        self.items = items
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds)
        self.min_len = int(target_sr * min_seconds)
        self.use_augmentation = use_augmentation

    def _load_wav(self, path: str) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        else:
            audio = audio[:, 0]

        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(
                torch.from_numpy(audio)
            ).numpy()

        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        elif len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode="constant")

        if len(audio) < self.min_len:
            audio = np.pad(audio, (0, self.min_len - len(audio)), mode="constant")

        return audio.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_id = self.items[idx]
        audio = self._load_wav(path)
        aug_audio = augment_waveform(audio, self.target_sr) if self.use_augmentation else audio.copy()

        return {
            "input_values": audio,
            "aug_input_values": aug_audio,
            "labels": int(label_id),
            "path": path,
        }


@dataclass
class LaSCLCollator:
    processor: AutoFeatureExtractor
    sampling_rate: int = 16000

    def __call__(self, features):
        audio = [f["input_values"] for f in features]
        aug_audio = [f["aug_input_values"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        batch_a = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        batch_aug = self.processor(aug_audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)

        return {
            "input_values": batch_a["input_values"],
            "attention_mask": batch_a.get("attention_mask"),
            "aug_input_values": batch_aug["input_values"],
            "aug_attention_mask": batch_aug.get("attention_mask"),
            "labels": labels,
        }