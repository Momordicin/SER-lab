from dataclasses import dataclass
from typing import List


@dataclass
class LaSCLConfig:
    audio_model_name: str = "microsoft/wavlm-base-plus"
    text_model_name: str = "roberta-base"

    sampling_rate: int = 16000
    max_seconds: float = 6.0
    min_seconds: float = 0.2

    batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 10

    audio_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01

    temperature: float = 0.07
    lambda_ce: float = 0.5
    lambda_scl: float = 0.5
    lambda_label_div: float = 0.0   # enable later

    proj_dim: int = 256
    proj_hidden_dim: int = 512
    dropout: float = 0.1

    freeze_feature_encoder: bool = True
    freeze_text_encoder: bool = True

    use_class_weights: bool = True
    seed: int = 42

    # Start with the paper-style 4 classes
    label_texts: List[str] = None

    def __post_init__(self):
        if self.label_texts is None:
            self.label_texts = ["angry", "happy", "neutral", "sad"]