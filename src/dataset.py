"""
dataset.py
----------
PyTorch Dataset wrapper for the VAST stance detection benchmark.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


class StanceDataset(Dataset):
    """
    Tokenises (tweet, topic) pairs for BERT-based stance detection.

    Expected CSV columns:
        - Tweet    : raw tweet text
        - Target 1 : topic / target string
        - label    : integer class label (set by LabelEncoder in preprocessing)

    Args:
        df        (pd.DataFrame): DataFrame with the columns above.
        tokenizer (BertTokenizer): Pre-loaded BERT tokenizer.
        max_len   (int): Maximum token length (default: 128).
    """

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 128):
        self.texts = df["Tweet"].values
        self.topics = df["Target 1"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            str(self.topics[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_and_preprocess(
    train_path: str,
    val_path: str,
    test_path: str,
    label_encoder,
) -> tuple:
    """
    Load CSV files, normalise stance labels, and apply label encoding.

    Args:
        train_path    (str): Path to training CSV.
        val_path      (str): Path to validation CSV.
        test_path     (str): Path to test CSV.
        label_encoder      : Fitted or unfitted sklearn LabelEncoder.

    Returns:
        Tuple of (train_df, val_df, test_df) with 'label' column added.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Normalise NONE → NEUTRAL across all splits
    for df in [train_df, val_df, test_df]:
        df["Stance 1"] = df["Stance 1"].replace("NONE", "NEUTRAL")

    train_df["label"] = label_encoder.fit_transform(train_df["Stance 1"])
    val_df["label"] = label_encoder.transform(val_df["Stance 1"])
    test_df["label"] = label_encoder.transform(test_df["Stance 1"])

    return train_df, val_df, test_df
