## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## Packages

```bash
pip install numpy torch torchaudio soundfile scikit-learn matplotlib transformers pandas umap-learn

or

pip install -r requirements.txt
```

## Download pretrained model

```bash
pip install -U huggingface-hub
pip install protobuf
mkdir -p models/wavlm-base-plus
hf download microsoft/wavlm-base-plus --local-dir models/wavlm-base-plus
```

## Train with

```bash
python3 lascl_ser.py train \
  --data_root dataset_split_loso_4cls \
  --out_dir lascl_ckpt_pilot \
  --batch_size 2 \
  --eval_batch_size 2 \
  --epochs 3 \
  --max_seconds 4.0 \
  --max_train_items 200 \
  --max_val_items 80
```

## Evaluate with

```bash
python3 lascl_ser.py train \
  --data_root dataset_split_loso_4cls \
  --out_dir lascl_ckpt_pilot \
  --batch_size 2 \
  --eval_batch_size 2 \
  --epochs 3 \
  --max_seconds 4.0 \
  --max_train_items 200 \
  --max_val_items 80
```

```bash
python3 lascl_ser.py eval \
  --data_root dataset_split_loso_4cls \
  --ckpt_path lascl_ckpt_pilot/best_model.pt \
  --split val \
  --mode classifier \
  --eval_batch_size 2 \
  --max_seconds 4.0
```
