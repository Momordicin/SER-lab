"""
IEMOCAP Data Preparation Script
Organizes raw IEMOCAP data into a flat folder format compatible with ser.py

Usage:
    python prepare_iemocap.py \
        --iemocap_root ./IEMOCAP_full_release \
        --output_dir   ./raw_wavs

Optional arguments:
    --merge_excited   Merge excited (exc) into happy (hap), recommended
    --sessions        Specify which sessions to use, defaults to all 1-5
"""

import os, re, shutil, argparse, collections

# Mapping from IEMOCAP emotion labels -> labels required by ser.py
EMO_MAP = {
    "ang": "ANG",
    "dis": "DIS",
    "fea": "FEA",
    "hap": "HAP",
    "neu": "NEU",
    "sad": "SAD",
    "exc": "HAP",   # excited merged into happy (takes effect when --merge_excited is enabled)
}

KEEP_LABELS = {"ang", "dis", "fea", "hap", "neu", "sad"}

def parse_emo_file(emo_txt_path: str):
    """
    Parses an EmoEvaluation txt file and returns a {utterance_id: emotion_str} dict.
    Example file format:
    [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
    """
    result = {}
    with open(emo_txt_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            # Match lines with timestamps
            m = re.match(r"\[[\d\.\s\-]+\]\s+(\S+)\s+(\w+)", line)
            if m:
                utt_id = m.group(1)
                emotion = m.group(2).lower()
                result[utt_id] = emotion
    return result

def prepare_iemocap(
    iemocap_root: str,
    output_dir: str,
    merge_excited: bool = True,
    sessions: list = None,
):
    if sessions is None:
        sessions = [1, 2, 3, 4, 5]

    os.makedirs(output_dir, exist_ok=True)

    keep_labels = set(KEEP_LABELS)
    if merge_excited:
        keep_labels.add("exc")

    stats = collections.Counter()
    skipped_emo = collections.Counter()
    copied = 0

    for sess_id in sessions:
        sess_dir = os.path.join(iemocap_root, f"Session{sess_id}")
        if not os.path.isdir(sess_dir):
            print(f"[WARN] Session{sess_id} not found, skipping.")
            continue

        emo_eval_dir = os.path.join(sess_dir, "dialog", "EmoEvaluation")
        wav_root     = os.path.join(sess_dir, "sentences", "wav")

        if not os.path.isdir(emo_eval_dir):
            print(f"[WARN] EmoEvaluation dir not found: {emo_eval_dir}")
            continue

        # Iterate over all EmoEvaluation txt files
        for emo_file in sorted(os.listdir(emo_eval_dir)):
            if not emo_file.endswith(".txt"):
                continue

            dialog_name = os.path.splitext(emo_file)[0]   # e.g. Ses01F_impro01
            emo_txt_path = os.path.join(emo_eval_dir, emo_file)
            utt_emo = parse_emo_file(emo_txt_path)

            dialog_wav_dir = os.path.join(wav_root, dialog_name)
            if not os.path.isdir(dialog_wav_dir):
                continue

            for utt_id, emotion in utt_emo.items():
                if emotion not in keep_labels:
                    skipped_emo[emotion] += 1
                    continue

                # Source wav path
                src_wav = os.path.join(dialog_wav_dir, f"{utt_id}.wav")
                if not os.path.isfile(src_wav):
                    continue

                # Target label
                tag = EMO_MAP.get(emotion, emotion.upper())

                # New filename: original name + emotion label, e.g. Ses01F_impro01_F000_NEU.wav
                new_name = f"{utt_id}_{tag}.wav"
                dst_wav  = os.path.join(output_dir, new_name)

                shutil.copy2(src_wav, dst_wav)
                stats[tag] += 1
                copied += 1

    # Summary report
    print(f"\n{'='*45}")
    print(f"  IEMOCAP data preparation complete")
    print(f"  Output directory: {output_dir}")
    print(f"  Total files copied: {copied}")
    print(f"\n  Count per emotion:")
    for lab in sorted(stats):
        print(f"    {lab:6s}  {stats[lab]}")
    if skipped_emo:
        print(f"\n  Skipped emotion labels (not in target set):")
        for lab, cnt in sorted(skipped_emo.items()):
            print(f"    {lab:6s}  {cnt}")
    print(f"{'='*45}\n")
    print("Next step: run ser.py split to partition the dataset")
    print(f"  python ser.py split --source_dir {output_dir} --target_dir ./dataset_split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IEMOCAP for ser.py")
    parser.add_argument("--iemocap_root", required=True, help="Root directory of IEMOCAP_full_release")
    parser.add_argument("--output_dir",   default="./raw_wavs", help="Output flat wav folder")
    parser.add_argument("--merge_excited", action="store_true", default=True,
                        help="Merge excited into happy (enabled by default)")
    parser.add_argument("--no_merge_excited", dest="merge_excited", action="store_false")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1,2,3,4,5],
                        help="Which sessions to use, default: 1 2 3 4 5")
    args = parser.parse_args()

    prepare_iemocap(
        iemocap_root=args.iemocap_root,
        output_dir=args.output_dir,
        merge_excited=args.merge_excited,
        sessions=args.sessions,
    )