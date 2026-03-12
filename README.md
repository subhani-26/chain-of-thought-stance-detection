# Chain-of-Thought Prompting for Zero-Shot Stance Detection (EZSD-CP)

A PyTorch implementation of **EZSD-CP** — a BERT-based model that uses chain-of-thought inspired prompt generation and gated MLP (gMLP) blocks for **zero-shot stance detection** on the VAST benchmark.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
  - [Evaluation (Inference Only)](#evaluation-inference-only)
  - [Training from Scratch](#training-from-scratch)
- [Results & Visualisations](#results--visualisations)
- [Citation](#citation)

---

## Overview

Stance detection identifies whether a piece of text expresses **FAVOR**, **AGAINST**, or **NEUTRAL** sentiment towards a given topic. The **zero-shot** setting is particularly challenging — the model must generalise to topics unseen during training.

This project implements:

- A **PromptGenerator** that creates soft prompt embeddings from BERT's CLS token
- A **gMLP** (Gated MLP) encoder for feature transformation
- Pairwise interaction features (difference + element-wise product) for richer representation
- Full evaluation pipeline with per-class and macro/micro F1 metrics
- Visualisation suite with publication-ready plots

---

## Model Architecture

```
Input: [Tweet] [SEP] [Target Topic]
         │
     BERT Encoder (bert-base-uncased)
         │
     CLS Embedding  [B, 768]
         │
   ┌─────┴──────┐
   │  Prompt    │
   │ Generator  │  → Prompt [B, 128]
   └─────┬──────┘
         │  cat([CLS, Prompt]) → [B, 896]
         │
    ┌────┴────┐
    │  gMLP   │  → v_c [B, 256]
    │  gMLP   │  → v_e [B, 256]
    └────┬────┘
         │
    [v_c, v_e, |v_c - v_e|, v_c * v_e]  → [B, 1024]
         │
    Linear Classifier  →  3 classes
```

---

## Project Structure

```
chain-of-thought-stance-detection/
│
├── main.py                  # Inference / evaluation entry-point
├── train.py                 # Training entry-point
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── model.py             # EZSD_CP_Model, PromptGenerator, gMLP
│   ├── dataset.py           # StanceDataset, load_and_preprocess
│   ├── evaluate.py          # run_inference, compute_metrics
│   └── visualize.py         # All plotting functions
│
├── data/
│   └── README.md            # Instructions for obtaining dataset & weights
│
├── outputs/                 # Generated plots saved here
│
└── notebooks/               # Original Colab notebook (reference only)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/chain-of-thought-stance-detection.git
cd chain-of-thought-stance-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Install the CUDA-compatible PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Data Setup

Place the following files in the `data/` directory:

| File | Description |
|---|---|
| `VAST_train_11k.csv` | Training split |
| `VAST_val.csv` | Validation split |
| `VAST_test.csv` | Test split |
| `stance_classification_model.pth` | Pre-trained weights (for eval only) |

See [`data/README.md`](data/README.md) for dataset download instructions.

**Required CSV columns:** `Tweet`, `Target 1`, `Stance 1`  
Labels: `FAVOR` / `AGAINST` / `NONE` (automatically mapped to `NEUTRAL`)

---

## Usage

### Evaluation (Inference Only)

Run evaluation on the test set using pre-trained weights:

```bash
python main.py \
    --train_csv  data/VAST_train_11k.csv \
    --val_csv    data/VAST_val.csv \
    --test_csv   data/VAST_test.csv \
    --model_path data/stance_classification_model.pth \
    --batch_size 8 \
    --output_dir outputs
```

This will:
- Print the full classification report (Precision / Recall / F1 per class)
- Print Macro and Micro F1 scores
- Save all plots to `outputs/`

### Training from Scratch

```bash
python train.py \
    --train_csv  data/VAST_train_11k.csv \
    --val_csv    data/VAST_val.csv \
    --test_csv   data/VAST_test.csv \
    --epochs     10 \
    --batch_size 16 \
    --lr         2e-5 \
    --save_path  data/stance_classification_model.pth \
    --output_dir outputs
```

### All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_csv` | `data/VAST_train_11k.csv` | Path to training CSV |
| `--val_csv` | `data/VAST_val.csv` | Path to validation CSV |
| `--test_csv` | `data/VAST_test.csv` | Path to test CSV |
| `--model_path` | `data/stance_classification_model.pth` | Path to model weights |
| `--batch_size` | `8` | Batch size |
| `--max_len` | `128` | Max tokenisation length |
| `--lr` | `2e-5` | Learning rate (train only) |
| `--epochs` | `10` | Training epochs (train only) |
| `--output_dir` | `outputs` | Directory for saved plots |
| `--bert_model` | `bert-base-uncased` | HuggingFace BERT model name |

---

## Results & Visualisations

After running evaluation, the following plots are saved to `outputs/`:

| File | Description |
|---|---|
| `precision_recall_f1.png` | Grouped bar chart per class |
| `macro_micro_f1_comparison.png` | Macro vs Micro F1 bar chart |
| `classification_report_metrics.png` | Three-panel subplot |
| `class_distribution_trends.png` | Class distribution across splits |

---

## Citation

If you use this code, please cite the original VAST dataset paper:

```bibtex
@inproceedings{allaway-mckeown-2020-zero,
    title     = "Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations",
    author    = "Allaway, Emily and McKeown, Kathleen",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    year      = "2020",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2020.emnlp-main.717",
}
```

---

## License

This project is released under the MIT License.
