"""
visualize.py
------------
All plotting functions for stance detection results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_precision_recall_f1(precision, recall, f1, class_names, save_dir: str = "outputs"):
    """Grouped bar chart — Precision, Recall, F1 per class."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x = np.arange(len(class_names))
    bar_width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x,                   precision, width=bar_width, label="Precision", color="#4C72B0")
    plt.bar(x + bar_width,       recall,    width=bar_width, label="Recall",    color="#DD8452")
    plt.bar(x + 2 * bar_width,   f1,        width=bar_width, label="F1 Score",  color="#55A868")

    plt.xticks(x + bar_width, class_names, fontsize=12)
    plt.xlabel("Class", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title("Precision, Recall & F1-score per Class", fontsize=15)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    path = f"{save_dir}/precision_recall_f1.png"
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[Saved] {path}")


def plot_macro_micro_f1(macro_f1: float, micro_f1: float, save_dir: str = "outputs"):
    """Bar chart comparing Macro vs Micro F1."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Macro F1", "Micro F1"], [macro_f1, micro_f1], color=["#E07B39", "#3C9E5F"])
    for bar, val in zip(bars, [macro_f1, micro_f1]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=12)

    plt.ylim(0, 1.05)
    plt.ylabel("F1 Score", fontsize=13)
    plt.title("Macro vs Micro F1 Comparison", fontsize=14)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    path = f"{save_dir}/macro_micro_f1_comparison.png"
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[Saved] {path}")


def plot_classification_report_subplots(report: dict, class_names, save_dir: str = "outputs"):
    """Three-panel subplot for Precision, Recall, F1 per class."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1-score"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, metric in enumerate(metrics):
        scores = [report[label][metric] for label in class_names]
        sns.barplot(x=class_names, y=scores, palette="viridis", ax=axs[i])
        axs[i].set_title(f"{metric.capitalize()} per Class", fontsize=13)
        axs[i].set_ylim(0, 1.05)
        axs[i].set_xlabel("Stance", fontsize=12)
        if i == 0:
            axs[i].set_ylabel("Score", fontsize=12)

    fig.suptitle("Classification Report Metrics (Test Set)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = f"{save_dir}/classification_report_metrics.png"
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[Saved] {path}")


def plot_class_distribution_subplots(
    train_df, val_df, test_df, label_encoder, save_dir: str = "outputs"
):
    """Three-panel subplot — class distribution across train / val / test splits."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    datasets = [train_df, val_df, test_df]
    titles = ["Train Set", "Validation Set", "Test Set"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, (df, title) in enumerate(zip(datasets, titles)):
        counts = df["label"].value_counts().sort_index()
        labels = label_encoder.inverse_transform(counts.index)
        sns.barplot(x=labels, y=counts.values, palette="Set2", ax=axs[i])
        axs[i].set_title(title, fontsize=13)
        axs[i].set_xlabel("Stance", fontsize=12)
        if i == 0:
            axs[i].set_ylabel("Count", fontsize=12)
        axs[i].set_xticklabels(labels, rotation=45)

    fig.suptitle("Class Distribution per Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = f"{save_dir}/class_distribution_trends.png"
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[Saved] {path}")


def run_all_plots(metrics: dict, report: dict, class_names, train_df, val_df, test_df,
                  label_encoder, save_dir: str = "outputs"):
    """Convenience wrapper — generates every plot in one call."""
    plot_precision_recall_f1(
        metrics["precision"], metrics["recall"], metrics["f1_per_class"],
        class_names, save_dir
    )
    plot_macro_micro_f1(metrics["macro_f1"], metrics["micro_f1"], save_dir)
    plot_classification_report_subplots(report, class_names, save_dir)
    plot_class_distribution_subplots(train_df, val_df, test_df, label_encoder, save_dir)
