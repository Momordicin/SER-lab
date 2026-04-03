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
    --out_dir ./ser_ckpt ^
    --epochs 10 ^
    --batch_size 8
```

### Benchmark Evaluation
```bash
python oldser.py test ^
    --data_root ./dataset_split ^
    --ckpt_dir ./ser_ckpt ^
    --preds_csv ./predictions.csv ^
    --report_txt ./report.txt
```

## SER with wavlm-base-plus  
### wavlm-base-plus Training
```bash
set TRANSFORMERS_OFFLINE=1 && python ser_wavlm.py train ^
  --data_root dataset_split ^
  --model_name models\wavlm-base-plus ^
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
### Train 
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

### Evaluate with  

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
