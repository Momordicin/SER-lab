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
hf download facebook/wav2vec2-base --local-dir models/wav2vec2-base
hf download microsoft/wavlm-base-plus --local-dir models/wavlm-base-plus
# huggingface-cli download microsoft/wavlm-base-plus
```

## Benchmark SER with wav2vec2  
### Dataset Preparation on IEMOCAP  
```bash
python prepare_iemocap.py ^
--iemocap_root ./IEMOCAP_full_release ^
--output_dir   ./raw_wavs

python oldser.py inspect --source_dir ./raw_wavs

python oldser.py split ^
    --source_dir ./raw_wavs ^
    --target_dir ./dataset_split
```

### Benchmark Training
```bash
python oldser.py train ^
    --data_root ./dataset_split ^
    --model_name models/wav2vec2-base ^
    --out_dir ./_ser_ckpt/ser_ckpt ^
    --epochs 10 ^
    --batch_size 8
```

```bash
python oldser.py train ^
    --labels ANG HAP NEU SAD ^
    --data_root ./dataset_split ^
    --model_name models/wav2vec2-base ^
    --out_dir ./_ser_ckpt/ser_ckpt_4class ^
    --epochs 10 ^
    --batch_size 8
```
### Benchmark Evaluation
```bash
python oldser.py test ^
    --data_root ./dataset_split ^
    --model_name models/wav2vec2-base ^
    --ckpt_dir ./_ser_ckpt/ser_ckpt ^ 
    --preds_csv ./results/predictions.csv ^
    --report_txt ./results/report.txt
```

```bash
python oldser.py test ^
    --data_root ./dataset_split ^
    --ckpt_dir ./_ser_ckpt/ser_ckpt_4class ^
    --batch_size 8 ^
    --preds_csv ./results/preds_4class.csv ^
    --report_txt ./results/report_4class.txt
```

### Zero-shot evaluation on RAVDESS
```bash
python cross_dataset_eval.py ^
    --ravdess_dir ravdess_audio ^
    --ckpt_dir ser_wavlm_ckpt ^
    --out_csv cross_dataset_preds_4class.csv
```

## SER with wavlm-base-plus  
### wavlm-base-plus Training
```bash
set TRANSFORMERS_OFFLINE=1 && python ser_wavlm.py train ^
  --data_root dataset_split ^
  --model_name models/wavlm-base-plus ^
  --out_dir ser_wavlm_ckpt ^
  --batch_size 8 ^
  --epochs 15 ^
  --lr 1e-5
```

### wavlm-base-plus Evaluation
```bash
python ser_wavlm.py test ^
  --data_root dataset_split ^
  --ckpt_dir ser_wavlm_ckpt ^
  --preds_csv wavlm_preds.csv ^
  --report_txt wavlm_report.txt
```

### wavlm-base-plus Cross-dataset Evaluation
```bash
python cross_dataset_eval.py --ravdess_dir ravdess_audio --ckpt_dir ser_wavlm_ckpt
```

### embedding visualization (optional)  
```bash
python visualize_embeddings.py ^
--data_root dataset_split ^
--ckpt_dir ser_wavlm_ckpt ^
--split test ^
--method tsne ^
--out_dir embed_vis
```

## Lascl pipeline  
### Training 
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

### Evaluation  
```bash
python3 lascl_ser.py eval \
  --data_root dataset_split_loso_4cls \
  --ckpt_path lascl_ckpt_pilot/best_model.pt \
  --split val \
  --mode classifier \
  --eval_batch_size 2 \
  --max_seconds 4.0
```
