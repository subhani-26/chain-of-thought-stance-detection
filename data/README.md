# data/

Place the following files in this directory before running training or evaluation:

| File | Description |
|---|---|
| `VAST_train_11k.csv` | VAST training split (~11k samples) |
| `VAST_val.csv` | VAST validation split |
| `VAST_test.csv` | VAST test split |
| `stance_classification_model.pth` | Pre-trained model weights (optional for inference) |

## Required CSV Columns

Each CSV file must contain **at minimum** these columns:

| Column | Type | Description |
|---|---|---|
| `Tweet` | str | Raw tweet text |
| `Target 1` | str | Stance target / topic |
| `Stance 1` | str | Label: `FAVOR`, `AGAINST`, or `NONE` / `NEUTRAL` |

## Obtaining the VAST Dataset

The VAST (Varied Stance Topics) dataset is available at:

- **Paper**: [Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations](https://aclanthology.org/2020.emnlp-main.717/)
- **GitHub**: https://github.com/emilyallaway/zero-shot-stance

## Model Weights

Pre-trained weights (`stance_classification_model.pth`) can be:
- Trained from scratch using `python train.py`
- Downloaded from the project release page (if provided)

> ⚠️ Large binary files (`.pth`, `.csv`) are excluded from Git via `.gitignore`.  
> Use [Git LFS](https://git-lfs.github.com/) or host weights on HuggingFace Hub / Google Drive.
