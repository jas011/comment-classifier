import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TARGET_COLS = ["real_query", "appreciation", "bad_experience", "noise"]


def load_jsonl_records(file_paths: List[str]) -> List[Dict]:
    valid_rows: List[Dict] = []
    skipped_count = 0

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty (0 bytes): {path}")

        with path.open("r", encoding="utf-8-sig") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    skipped_count += 1
                    print(
                        f"Warning: Skipping malformed JSON at {path}:{line_num} ({exc}): {line[:100]!r}"
                    )
                    continue

                if isinstance(row, dict):
                    valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid JSON records found across the provided files.")

    df = pd.DataFrame(valid_rows)
    if "text" not in df.columns:
        raise ValueError("Dataset records are missing the 'text' column.")

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
            .astype(float)
        )

    if len(df) < 10:
        raise ValueError(
            f"Need at least 10 valid records to train, found {len(df)}."
        )

    print(
        f"Loaded {len(df)} valid records via pandas (skipped {skipped_count} malformed line(s))."
    )
    return [
        {
            "text": row["text"],
            "labels": [float(row[col]) for col in TARGET_COLS],
        }
        for _, row in df.iterrows()
    ]
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds_all.append(probs)
            targets_all.append(labels.cpu().numpy())

    preds = np.vstack(preds_all)
    targets = np.vstack(targets_all)
    mae = float(np.mean(np.abs(preds - targets)))
    return {
        "val_loss": total_loss / len(loader.dataset),
        "val_mae": mae,
    }


def train(
    data_files: List[str],
    output_dir: str = "./distilbert_comment_model",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 192,
    val_split: float = 0.15,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_jsonl_records(data_files)
    hf_dataset = Dataset.from_list(records).shuffle(seed=seed)

    test_size = max(1, int(round(len(hf_dataset) * val_split)))
    test_size = min(test_size, len(hf_dataset) - 1)
    split = hf_dataset.train_test_split(test_size=test_size, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["labels"]
        return enc

    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_COLS),
        problem_type="multi_label_classification",
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | Val MAE: {metrics['val_mae']:.4f}"
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
            with (out_path / "training_meta.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "target_cols": TARGET_COLS,
                        "max_length": max_length,
                        "best_val_loss": best_val_loss,
                        "val_mae": metrics["val_mae"],
                        "sources": data_files,
                    },
                    f,
                    indent=2,
                )
            print(f"  -> Saved improved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DistilBERT for soft comment scoring."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more JSONL dataset paths (e.g., youtube_silver_labels.jsonl amazon_silver_labels.jsonl)",
    )
    parser.add_argument("--output-dir", default="./distilbert_comment_model")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    train(
        data_files=args.data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )