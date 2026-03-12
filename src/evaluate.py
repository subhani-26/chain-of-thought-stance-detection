"""
evaluate.py
-----------
Evaluation utilities: inference loop + metric computation.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def run_inference(model, loader: DataLoader, device: torch.device) -> tuple:
    """
    Run model inference over a DataLoader.

    Args:
        model  : Trained EZSD_CP_Model.
        loader : DataLoader for the evaluation split.
        device : torch.device (cpu / cuda).

    Returns:
        (preds, labels) — lists of integer predictions and ground-truth labels.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def compute_metrics(labels, preds, class_names) -> dict:
    """
    Compute full classification metrics.

    Args:
        labels      : Ground-truth integer labels.
        preds       : Predicted integer labels.
        class_names : List of human-readable class names.

    Returns:
        dict with keys: report, macro_f1, micro_f1, precision, recall, f1_per_class
    """
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )

    print("\n" + "=" * 55)
    print("  Classification Report")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=class_names))
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Micro F1 : {micro_f1:.4f}")
    print("=" * 55 + "\n")

    return {
        "report": report,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1_per_class,
    }
