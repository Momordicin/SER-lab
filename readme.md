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
