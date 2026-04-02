import random
import numpy as np
import torch
import torchaudio


def add_noise(audio: np.ndarray, noise_scale: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(audio)).astype(np.float32)
    return (audio + noise_scale * noise).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int, steps: float = 2.0) -> np.ndarray:
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    y = torchaudio.functional.pitch_shift(x, sr, n_steps=steps)
    return y.squeeze(0).numpy().astype(np.float32)


def simple_reverb_like(audio: np.ndarray, sr: int) -> np.ndarray:
    x = torch.tensor(audio, dtype=torch.float32)
    y = torchaudio.functional.lowpass_biquad(x, sample_rate=sr, cutoff_freq=3000)
    return y.numpy().astype(np.float32)


def augment_waveform(audio: np.ndarray, sr: int) -> np.ndarray:
    op = random.choice(["noise", "pitch", "reverb", "identity"])

    if op == "noise":
        return add_noise(audio)
    if op == "pitch":
        try:
            return pitch_shift(audio, sr, steps=random.choice([-2.0, 2.0]))
        except Exception:
            return audio.astype(np.float32)
    if op == "reverb":
        try:
            return simple_reverb_like(audio, sr)
        except Exception:
            return audio.astype(np.float32)

    return audio.astype(np.float32)