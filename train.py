"""
train.py
--------
Fine-tuning script for the EZSD-CP stance detection model.

Usage:
    python train.py \
        --train_csv  data/VAST_train_11k.csv \
        --val_csv    data/VAST_val.csv \
        --test_csv   data/VAST_test.csv \
        --epochs     10 \
        --batch_size 16 \
        --lr         2e-5 \
        --output_dir outputs
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

from src.model import EZSD_CP_Model
from src.dataset import StanceDataset, load_and_preprocess
from src.evaluate import run_inference, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train EZSD-CP model")
    parser.add_argument("--train_csv",  type=str, default="data/VAST_train_11k.csv")
    parser.add_argument("--val_csv",    type=str, default="data/VAST_val.csv")
    parser.add_argument("--test_csv",   type=str, default="data/VAST_test.csv")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_len",    type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--save_path", type=str, default="data/stance_classification_model.pth",
        help="Where to save the trained model weights"
    )
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    # Data
    label_encoder = LabelEncoder()
    train_df, val_df, test_df = load_and_preprocess(
        args.train_csv, args.val_csv, args.test_csv, label_encoder
    )
    class_names = list(label_encoder.classes_)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_dataset = StanceDataset(train_df, tokenizer, args.max_len)
    val_dataset = StanceDataset(val_df, tokenizer, args.max_len)
    test_dataset = StanceDataset(test_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model, optimiser, loss
    model = EZSD_CP_Model(bert_model_name=args.bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    print(f"\n[Train] Starting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✓ Best model saved to '{args.save_path}' (val_acc={val_acc:.4f})")

    # Final test evaluation
    print("\n[Eval] Running final evaluation on test set...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    preds, labels = run_inference(model, test_loader, device)
    compute_metrics(labels, preds, class_names)
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
