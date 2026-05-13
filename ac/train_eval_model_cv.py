"""Run 5-fold cross-validation for the assertion classifier.

This is the Table 4 reproduction script. It pools
`ac/data/{train,val,test}_expanded.csv`, fine-tunes a fresh RoBERTa-PM
classifier in each fold with the `<entity>` special token added, and reports
loss, accuracy, macro-F1, and per-class metrics.

Use `train_ac_model.py` when you need the single deployment checkpoint. The
CV defaults are smaller than the deployment defaults (batch size 32 vs 8, max
length 128 vs 512) to keep 5-fold runs tractable; this matches the setup used
to produce Table 4. Pass `--no-wandb` for offline or headless runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bars
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
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
        add_log_level,
        configure_logging,
        progress_disabled,
        require_columns,
        require_directory,
        require_file,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("train_eval_model_cv")
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb-project", default="assertion-classification")
    parser.add_argument("--wandb-run-name", default="CrossVal_5fold")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)

    require_directory(args.base_model, "Base RoBERTa-PM model directory")
    require_file(args.base_model / "config.json", "Base RoBERTa-PM config")

    train_df = read_split(args.data_dir, "train")
    test_df = read_split(args.data_dir, "test")
    val_df = read_split(args.data_dir, "val")
    all_df = pd.concat([train_df, test_df, val_df])

    label_encoder = LabelEncoder()
    all_df["label"] = label_encoder.fit_transform(all_df["assertion"])
    class_names = [str(c) for c in label_encoder.classes_]
    LOGGER.info("Label vocabulary: %s", class_names)

    chosen_model = str(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(chosen_model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<entity>"]})

    config = AutoConfig.from_pretrained(chosen_model, num_labels=len(class_names))

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    gradient_accumulation_steps = args.gradient_accumulation_steps
    patience = args.patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on %s", device)

    n_splits = args.n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    use_wandb = not args.no_wandb and not os.environ.get("WANDB_DISABLED")
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_name": chosen_model,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "n_splits": n_splits,
                "max_length": args.max_length,
                "seed": args.seed,
            },
        )
    else:
        wandb = None  # type: ignore

    texts = all_df["text"].values
    labels = all_df["label"].values

    fold_val_losses = []
    fold_val_accuracies = []
    fold_val_f1s = []

    class_metrics = {
        class_name: {"precision": [], "recall": [], "f1-score": []}
        for class_name in class_names
    }

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

    disable_progress = progress_disabled(LOGGER)
    if disable_progress:
        disable_progress_bars()

    fold_tmp_dir = args.data_dir / "_cv_fold_checkpoints"
    fold_tmp_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_index, val_index) in enumerate(skf.split(texts, labels)):
        LOGGER.info("Fold %s/%s", fold_idx + 1, n_splits)

        train_texts, val_texts = texts[train_index], texts[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        encoded_train = train_dataset.map(preprocess_function, batched=True)
        encoded_val = val_dataset.map(preprocess_function, batched=True)

        encoded_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        encoded_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(encoded_val, batch_size=batch_size, shuffle=False)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        fold_model = AutoModelForSequenceClassification.from_pretrained(chosen_model, config=config)
        fold_model.resize_token_embeddings(len(tokenizer))
        fold_model.to(device)

        optimizer = AdamW(fold_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * epochs
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = 0
        early_stop_counter = 0
        best_model_path = fold_tmp_dir / f"fold_{fold_idx + 1}_best_model.pt"

        for epoch in range(epochs):
            LOGGER.info("Epoch %s/%s for fold %s", epoch + 1, epochs, fold_idx + 1)

            fold_model.train()
            train_loss = 0.0
            train_steps = 0
            train_preds = []
            train_labels_list = []

            for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["label"].to(device)

                outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels_t)

                loss = loss / gradient_accumulation_steps
                loss.backward()
                train_loss += loss.item() * gradient_accumulation_steps

                preds = torch.argmax(logits, dim=-1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels_t.cpu().numpy())

                if (train_steps + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                train_steps += 1

            avg_train_loss = train_loss / train_steps
            train_accuracy = accuracy_score(train_labels_list, train_preds)
            train_f1 = f1_score(train_labels_list, train_preds, average="macro")

            fold_model.eval()
            val_loss = 0
            val_steps = 0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_t = batch["label"].to(device)

                    outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    loss = loss_fn(logits, labels_t)
                    val_loss += loss.item()
                    val_steps += 1

                    preds = torch.argmax(logits, dim=-1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels_t.cpu().numpy())

            avg_val_loss = val_loss / val_steps
            val_accuracy = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average="macro")

            if wandb is not None:
                wandb.log(
                    {
                        f"fold_{fold_idx + 1}_epoch": epoch + 1,
                        f"fold_{fold_idx + 1}_train_loss": avg_train_loss,
                        f"fold_{fold_idx + 1}_train_accuracy": train_accuracy,
                        f"fold_{fold_idx + 1}_train_f1_macro": train_f1,
                        f"fold_{fold_idx + 1}_val_loss": avg_val_loss,
                        f"fold_{fold_idx + 1}_val_accuracy": val_accuracy,
                        f"fold_{fold_idx + 1}_val_f1_macro": val_f1,
                    }
                )

            LOGGER.info(
                "Fold %s epoch %s: train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f",
                fold_idx + 1,
                epoch + 1,
                avg_train_loss,
                train_f1,
                avg_val_loss,
                val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                early_stop_counter = 0
                torch.save(fold_model.state_dict(), best_model_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    LOGGER.info("Early stopping at epoch %s for fold %s", epoch + 1, fold_idx + 1)
                    break

        fold_model.load_state_dict(torch.load(best_model_path))
        fold_model.eval()

        val_loss = 0
        val_steps = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in DataLoader(encoded_val, batch_size=batch_size):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["label"].to(device)

                outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, labels_t)
                val_loss += loss.item()
                val_steps += 1

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels_t.cpu().numpy())

        final_val_loss = val_loss / val_steps
        final_val_accuracy = accuracy_score(val_true, val_preds)
        final_val_f1 = f1_score(val_true, val_preds, average="macro")

        fold_val_losses.append(final_val_loss)
        fold_val_accuracies.append(final_val_accuracy)
        fold_val_f1s.append(final_val_f1)

        report_dict = classification_report(val_true, val_preds, target_names=class_names, output_dict=True)
        for class_name in class_names:
            class_metrics[class_name]["precision"].append(report_dict[class_name]["precision"])
            class_metrics[class_name]["recall"].append(report_dict[class_name]["recall"])
            class_metrics[class_name]["f1-score"].append(report_dict[class_name]["f1-score"])

        LOGGER.info(
            "Fold %s final: val_loss=%.4f val_accuracy=%.4f val_f1_macro=%.4f",
            fold_idx + 1,
            final_val_loss,
            final_val_accuracy,
            final_val_f1,
        )

    mean_loss = np.mean(fold_val_losses)
    std_loss = np.std(fold_val_losses)
    mean_acc = np.mean(fold_val_accuracies)
    std_acc = np.std(fold_val_accuracies)
    mean_f1 = np.mean(fold_val_f1s)
    std_f1 = np.std(fold_val_f1s)

    LOGGER.info("5-fold aggregate val loss: %.4f +/- %.4f", mean_loss, std_loss)
    LOGGER.info("5-fold aggregate val accuracy: %.4f +/- %.4f", mean_acc, std_acc)
    LOGGER.info("5-fold aggregate val macro F1: %.4f +/- %.4f", mean_f1, std_f1)

    for class_name in class_names:
        prec_mean = np.mean(class_metrics[class_name]["precision"])
        prec_std = np.std(class_metrics[class_name]["precision"])
        rec_mean = np.mean(class_metrics[class_name]["recall"])
        rec_std = np.std(class_metrics[class_name]["recall"])
        f1_mean = np.mean(class_metrics[class_name]["f1-score"])
        f1_std = np.std(class_metrics[class_name]["f1-score"])

        LOGGER.info(
            "Class %s: precision %.3f +/- %.3f, recall %.3f +/- %.3f, F1 %.3f +/- %.3f",
            class_name,
            prec_mean,
            prec_std,
            rec_mean,
            rec_std,
            f1_mean,
            f1_std,
        )

    if wandb is not None:
        wandb.log(
            {
                "cv_mean_val_loss": mean_loss,
                "cv_std_val_loss": std_loss,
                "cv_mean_val_accuracy": mean_acc,
                "cv_std_val_accuracy": std_acc,
                "cv_mean_val_f1": mean_f1,
                "cv_std_val_f1": std_f1,
            }
        )

        for class_name in class_names:
            wandb.log(
                {
                    f"cv_{class_name}_precision_mean": np.mean(class_metrics[class_name]["precision"]),
                    f"cv_{class_name}_precision_std": np.std(class_metrics[class_name]["precision"]),
                    f"cv_{class_name}_recall_mean": np.mean(class_metrics[class_name]["recall"]),
                    f"cv_{class_name}_recall_std": np.std(class_metrics[class_name]["recall"]),
                    f"cv_{class_name}_f1_mean": np.mean(class_metrics[class_name]["f1-score"]),
                    f"cv_{class_name}_f1_std": np.std(class_metrics[class_name]["f1-score"]),
                }
            )

        wandb.finish()


if __name__ == "__main__":
    main()
