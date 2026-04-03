"""
cross_dataset_eval.py
---------------------
Zero-shot evaluation of a trained WavLM/Wav2Vec2 checkpoint on RAVDESS.

Step 1: Download RAVDESS Audio_Speech_Actors_01-24.zip from
        https://zenodo.org/record/1188976
        Unzip to a folder, e.g. ravdess_audio\

Step 2: Run:
    python cross_dataset_eval.py ^
        --ravdess_dir ravdess_audio ^
        --ckpt_dir    ser_wavlm_ckpt ^
        --out_csv     cross_dataset_preds.csv

RAVDESS filename format:
  03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav
  emotion codes: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
"""

import os, glob, argparse
import numpy as np
import torch
import soundfile as sf
import torchaudio
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoProcessor, AutoFeatureExtractor, WavLMForSequenceClassification
from transformers import Wav2Vec2ForSequenceClassification, WavLMForSequenceClassification

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

# ── RAVDESS emotion code → label (only keep classes that overlap with IEMOCAP)
RAVDESS_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    # "06": "fear",   # RAVDESS "fearful" → IEMOCAP "fear"
    # "02": "calm",      # no IEMOCAP equivalent, skipped
    # "07": "disgust",   # present in IEMOCAP but usually skipped
    # "08": "surprised", # no IEMOCAP equivalent, skipped
}

def load_model(ckpt_dir):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        model_type = json.load(f).get("model_type", "")
    if model_type == "wavlm":
        return WavLMForSequenceClassification.from_pretrained(ckpt_dir).eval()
    else:
        return Wav2Vec2ForSequenceClassification.from_pretrained(ckpt_dir).eval()


def parse_ravdess(path):
    """Return label string or None if emotion not in RAVDESS_MAP."""
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.split("-")
    if len(parts) < 3:
        return None
    return RAVDESS_MAP.get(parts[2], None)

def load_audio(path, target_sr=16000, max_seconds=6.0):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(
            torch.from_numpy(audio)).numpy()
    max_len = int(target_sr * max_seconds)
    if len(audio) > max_len:
        audio = audio[:max_len]
    elif len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    return audio

def evaluate(ravdess_dir, ckpt_dir, out_csv, batch_size=16, sr=16000, max_seconds=6.0):
    # ── collect files
    wavs = glob.glob(os.path.join(ravdess_dir, "**", "*.wav"), recursive=True)
    items = []
    for p in sorted(wavs):
        lab = parse_ravdess(p)
        if lab is not None:
            items.append((p, lab))

    if not items:
        raise RuntimeError("No matching RAVDESS files found. Check --ravdess_dir.")
    print(f"Found {len(items)} files with mapped labels.")

    # ── load checkpoint
    processor = AutoFeatureExtractor.from_pretrained(ckpt_dir)
    model = load_model(ckpt_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── build label mapping from checkpoint config
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    # filter to classes the checkpoint knows
    valid_items = [(p, lab) for p, lab in items if lab in label2id]
    skipped     = len(items) - len(valid_items)
    if skipped:
        print(f"Skipped {skipped} files (label not in checkpoint): "
              f"{set(l for _,l in items) - set(label2id.keys())}")
    print(f"Evaluating {len(valid_items)} files across labels: "
          f"{sorted(set(l for _,l in valid_items))}")

    y_true, y_pred, paths_out = [], [], []

    for i in range(0, len(valid_items), batch_size):
        batch = valid_items[i:i+batch_size]
        audios = [load_audio(p, sr, max_seconds) for p, _ in batch]
        labs   = [label2id[l] for _, l in batch]

        enc = processor(audios, sampling_rate=sr, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        y_true.extend(labs)
        y_pred.extend(preds.tolist())
        paths_out.extend([p for p, _ in batch])

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {i+len(batch)}/{len(valid_items)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── results
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    present_ids   = sorted(set(y_true) | set(y_pred))
    present_names = [id2label[i] for i in present_ids]

    print(f"\nCROSS-DATASET RESULTS (RAVDESS → {os.path.basename(ckpt_dir)})")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1m:.4f}")
    print(f"F1 weighted: {f1w:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred,
                                 labels=present_ids, target_names=present_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=present_ids))

    # ── save CSV
    true_names = [id2label[i] for i in y_true]
    pred_names = [id2label[i] for i in y_pred]
    if HAS_PD:
        import pandas as pd
        pd.DataFrame({"path": paths_out,
                      "y_true": true_names,
                      "y_pred": pred_names}).to_csv(out_csv, index=False)
    else:
        with open(out_csv, "w") as f:
            f.write("path,y_true,y_pred\n")
            for p, t, pr in zip(paths_out, true_names, pred_names):
                f.write(f"{p},{t},{pr}\n")
    print(f"\nSaved predictions to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess_dir", required=True)
    parser.add_argument("--ckpt_dir",    default="ser_wavlm_ckpt")
    parser.add_argument("--out_csv",     default="cross_dataset_preds.csv")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--sr",          type=int,   default=16000)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    args = parser.parse_args()
    evaluate(args.ravdess_dir, args.ckpt_dir, args.out_csv,
             args.batch_size, args.sr, args.max_seconds)

if __name__ == "__main__":
    main()