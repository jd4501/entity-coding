"""Fine-tune the deployment assertion classifier.

This script trains the AC model that `ner/extract_entities.py` loads at
inference time. It reads `ac/data/{train,val,test}_expanded.csv`, fine-tunes
RoBERTa-PM with the `<entity>` special token added, and exports a HuggingFace
model directory plus `label_encoder.joblib` under `data/models/ac_model/`.

Use `train_eval_model_cv.py` for Table 4 cross-validation metrics. Use this
script when you need the single deployment checkpoint. Pass `--no-wandb` for
offline or headless runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, disable_progress_bars
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from common import (
        AC_DATA_DIR,
        BASE_ROBERTA_PM_DIR,
        MODELS_DIR,
        add_log_level,
        configure_logging,
        progress_disabled,
        require_columns,
        require_directory,
        require_file,
    )
except ModuleNotFoundError:
    from ac.common import (
        AC_DATA_DIR,
        BASE_ROBERTA_PM_DIR,
        MODELS_DIR,
        add_log_level,
        configure_logging,
        progress_disabled,
        require_columns,
        require_directory,
        require_file,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("train_ac_model")
REQUIRED_COLUMNS = {"text", "assertion"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = require_file(data_dir / f"{split}_expanded.csv", f"{split} AC split")
    df = pd.read_csv(path)
    require_columns(df.columns, REQUIRED_COLUMNS, path)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=AC_DATA_DIR)
    parser.add_argument("--base-model", type=Path, default=BASE_ROBERTA_PM_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "ac_model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb-project", default="assertion-classification")
    parser.add_argument("--wandb-run-name", default="AC_eval_model")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)

    require_directory(args.base_model, "Base RoBERTa-PM model directory")
    require_file(args.base_model / "config.json", "Base RoBERTa-PM config")

    train_df = read_split(args.data_dir, "train")
    val_df = read_split(args.data_dir, "val")
    test_df = read_split(args.data_dir, "test")

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["assertion"])
    val_df["label"] = label_encoder.transform(val_df["assertion"])
    test_df["label"] = label_encoder.transform(test_df["assertion"])
    class_names = [str(c) for c in label_encoder.classes_]
    LOGGER.info("Label vocabulary: %s", class_names)

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_df["text"].tolist(), "label": train_df["label"].tolist()}),
            "validation": Dataset.from_dict({"text": val_df["text"].tolist(), "label": val_df["label"].tolist()}),
            "test": Dataset.from_dict({"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<entity>"]})

    config = AutoConfig.from_pretrained(str(args.base_model), num_labels=len(class_names))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.base_model), config=config)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

    disable_progress = progress_disabled(LOGGER)
    if disable_progress:
        disable_progress_bars()

    encoded = dataset.map(preprocess, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(encoded["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(encoded["validation"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(encoded["test"], batch_size=args.batch_size, shuffle=False)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"].values),
        y=train_df["label"].values,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    model.to(device)
    LOGGER.info("Training on %s", device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    use_wandb = not args.no_wandb and not os.environ.get("WANDB_DISABLED")
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": warmup_steps,
                "model_name": str(args.base_model),
                "max_length": args.max_length,
                "seed": args.seed,
            },
        )
    else:
        wandb = None  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        LOGGER.info("Epoch %s/%s", epoch + 1, args.epochs)
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels_list, train_preds, average="macro")

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                val_loss += loss_fn(logits, labels).item()
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)

        LOGGER.info(
            "train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f",
            avg_train_loss,
            train_f1,
            avg_val_loss,
            val_f1,
            val_acc,
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_f1_macro": train_f1,
                    "val_loss": avg_val_loss,
                    "val_f1_macro": val_f1,
                    "val_accuracy": val_acc,
                }
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(str(args.output_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(args.output_dir))
            joblib.dump(label_encoder, args.output_dir / "label_encoder.joblib")
            LOGGER.info("Saved best model to %s with val_f1=%.4f", args.output_dir, val_f1)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                LOGGER.info("Early stopping at epoch %s", epoch + 1)
                break

    LOGGER.info("Reloading best checkpoint for test evaluation")
    model = AutoModelForSequenceClassification.from_pretrained(str(args.output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(args.output_dir))
    model.to(device)
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False, disable=disable_progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro")
    LOGGER.info("Test accuracy: %.4f", test_acc)
    LOGGER.info("Test macro F1: %.4f", test_f1)
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(test_true, test_preds, target_names=class_names, digits=3),
    )
    LOGGER.info("Confusion matrix:\n%s", confusion_matrix(test_true, test_preds))

    if wandb is not None:
        report = classification_report(test_true, test_preds, target_names=class_names, output_dict=True)
        wandb.log(
            {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
                **{f"test_f1_{class_name}": report[class_name]["f1-score"] for class_name in class_names},
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
