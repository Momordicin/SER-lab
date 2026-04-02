"""
visualize_embeddings.py
-----------------------
Extract WavLM embeddings from a trained checkpoint and visualize
emotional separability via t-SNE (and optionally UMAP).

Usage:
    python visualize_embeddings.py ^
        --data_root dataset_split ^
        --ckpt_dir  ser_wavlm_ckpt ^
        --split     test ^
        --method    tsne ^
        --out_dir   embed_vis
"""

import os, glob, argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from transformers import AutoProcessor, AutoFeatureExtractor, WavLMForSequenceClassification


# ── helpers (minimal, no dependency on ser_wavlm.py) ──────────────────────────

def is_real_wav(p):
    b = os.path.basename(p)
    return b.lower().endswith(".wav") and not b.startswith("._")

def list_wavs_recursive(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
            if is_real_wav(p)]

def get_labels_from_train(train_dir):
    labs = []
    for name in sorted(os.listdir(train_dir)):
        p = os.path.join(train_dir, name)
        if os.path.isdir(p) and list_wavs_recursive(p):
            labs.append(name.lower())
    if not labs:
        raise RuntimeError("No label subfolders in train split.")
    return labs

def load_items(split_dir, label2id):
    items = []
    for lab in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, lab)
        if not os.path.isdir(d): continue
        lid = label2id.get(lab.lower())
        if lid is None: continue
        for p in list_wavs_recursive(d):
            items.append((p, lid))
    return items

def load_audio(path, processor, target_sr=16000, max_seconds=6.0):
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


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(data_root, ckpt_dir, split="test",
                       sr_target=16000, max_seconds=6.0, batch_size=8):
    train_dir = os.path.join(data_root, "train")
    split_dir = os.path.join(data_root, split)

    labels     = get_labels_from_train(train_dir)
    label2id   = {lab: i for i, lab in enumerate(labels)}
    items      = load_items(split_dir, label2id)
    if not items:
        raise RuntimeError(f"No wav files found in split: {split_dir}")

    print(f"Found {len(items)} files in '{split}' split.")

    processor = AutoFeatureExtractor.from_pretrained(ckpt_dir)
    model     = WavLMForSequenceClassification.from_pretrained(ckpt_dir)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    all_emb, all_lab = [], []

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        audios = [load_audio(p, processor, sr_target, max_seconds)
                  for p, _ in batch_items]
        labs   = [lab for _, lab in batch_items]

        enc = processor(audios, sampling_rate=sr_target,
                        return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            # hidden states from last transformer layer, mean-pooled
            out = model.wavlm(**enc)
            # out.last_hidden_state: [B, T, D]
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()  # [B, D]

        all_emb.append(emb)
        all_lab.extend(labs)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  processed {i+len(batch_items)}/{len(items)}")

    embeddings  = np.vstack(all_emb)          # [N, 768]
    label_ids   = np.array(all_lab)           # [N]
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, label_ids, labels


# ── dimensionality reduction + plot ───────────────────────────────────────────

def plot_2d(coords, label_ids, label_names, title, out_path):
    n_classes = len(label_names)
    cmap      = cm.get_cmap("tab10", n_classes)
    colors    = [cmap(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(label_names):
        mask = label_ids == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=name, alpha=0.65, s=18)

    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def visualize(data_root, ckpt_dir, split="test", method="tsne",
              perplexity=30, n_neighbors=15, out_dir="embed_vis",
              sr_target=16000, max_seconds=6.0, batch_size=8):

    os.makedirs(out_dir, exist_ok=True)

    embeddings, label_ids, label_names = extract_embeddings(
        data_root, ckpt_dir, split, sr_target, max_seconds, batch_size)

    # save raw embeddings for later use
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, "labels.npy"), label_ids)
    print("Raw embeddings saved.")

    methods = [method] if method != "both" else ["tsne", "umap"]

    for m in methods:
        if m == "umap":
            if not HAS_UMAP:
                print("umap-learn not installed, skipping UMAP. pip install umap-learn")
                continue
            print("Running UMAP...")
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            coords  = reducer.fit_transform(embeddings)
            title   = f"UMAP — WavLM embeddings ({split})"
            fname   = f"umap_{split}.png"

        else:  # tsne
            print("Running t-SNE (this may take a minute)...")
            perp = min(perplexity, len(embeddings) - 1)
            reducer = TSNE(n_components=2, perplexity=perp,
                           random_state=42, max_iter=1000)
            coords  = reducer.fit_transform(embeddings)
            title   = f"t-SNE — WavLM embeddings ({split})"
            fname   = f"tsne_{split}.png"

        plot_2d(coords, label_ids, label_names, title,
                os.path.join(out_dir, fname))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WavLM Embedding Visualizer")
    parser.add_argument("--data_root",    default="dataset_split")
    parser.add_argument("--ckpt_dir",     default="ser_wavlm_ckpt")
    parser.add_argument("--split",        default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--method",       default="tsne",
                        choices=["tsne", "umap", "both"])
    parser.add_argument("--perplexity",   type=int,   default=30)
    parser.add_argument("--n_neighbors",  type=int,   default=15)
    parser.add_argument("--out_dir",      default="embed_vis")
    parser.add_argument("--sr_target",    type=int,   default=16000)
    parser.add_argument("--max_seconds",  type=float, default=6.0)
    parser.add_argument("--batch_size",   type=int,   default=8)
    args = parser.parse_args()

    visualize(
        data_root   = args.data_root,
        ckpt_dir    = args.ckpt_dir,
        split       = args.split,
        method      = args.method,
        perplexity  = args.perplexity,
        n_neighbors = args.n_neighbors,
        out_dir     = args.out_dir,
        sr_target   = args.sr_target,
        max_seconds = args.max_seconds,
        batch_size  = args.batch_size,
    )

if __name__ == "__main__":
    main()