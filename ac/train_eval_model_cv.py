import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

# Data
train_df = pd.read_csv('ac/data/train_expanded.csv')
test_df = pd.read_csv('ac/data/test_expanded.csv')
val_df = pd.read_csv('ac/data/val_expanded.csv')

all_df = pd.concat([train_df, test_df, val_df])

label_encoder = LabelEncoder()
all_df['label'] = label_encoder.fit_transform(all_df['assertion'])

class_names = label_encoder.classes_

# Setup model
CHOSEN_MODEL = "data/models/RoBERTa-base-PM-M3-Voc-distill-align-hf"

tokenizer = AutoTokenizer.from_pretrained(CHOSEN_MODEL, use_fast=True)

# Add special token <entity>
special_tokens_dict = {'additional_special_tokens': ['<entity>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

config = AutoConfig.from_pretrained(
    CHOSEN_MODEL,
    num_labels=len(class_names)
)

model = AutoModelForSequenceClassification.from_pretrained(
    CHOSEN_MODEL,
    config=config
)

# Resize token embeddings to match the new tokenizer
model.resize_token_embeddings(len(tokenizer))

# Hyperparameters
epochs = 12
batch_size = 32
learning_rate = 3e-5
weight_decay = 0.01
gradient_accumulation_steps = 1
patience = 4 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

wandb.init(
    project='assertion-classification',
    name='CrossVal_5fold',
    config={
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'model_name': CHOSEN_MODEL,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'n_splits': n_splits
    }
)

texts = all_df['text'].values
labels = all_df['label'].values

fold_val_losses = []
fold_val_accuracies = []
fold_val_f1s = []

class_metrics = {
    cname: {"precision": [], "recall": [], "f1-score": []}
    for cname in class_names
}

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

for fold_idx, (train_index, val_index) in enumerate(skf.split(texts, labels)):
    print(f"\n========== FOLD {fold_idx+1} / {n_splits} ==========")

    train_texts, val_texts = texts[train_index], texts[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset   = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    encoded_train = train_dataset.map(preprocess_function, batched=True)
    encoded_val   = val_dataset.map(preprocess_function, batched=True)

    encoded_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    encoded_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(encoded_val,   batch_size=batch_size, shuffle=False)

    # For class weighted loss
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    fold_model = AutoModelForSequenceClassification.from_pretrained(CHOSEN_MODEL, config=config)
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
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    early_stop_counter = 0
    best_model_path = f"fold_{fold_idx+1}_best_model.pt"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} (Fold {fold_idx+1})")

        # ----- TRAINING -----
        fold_model.train()
        train_loss = 0.0
        train_steps = 0
        train_preds = []
        train_labels_list = []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_t = batch['label'].to(device)

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
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        # ----- VALIDATION -----
        fold_model.eval()
        val_loss = 0
        val_steps = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_t = batch['label'].to(device)

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
        val_f1 = f1_score(val_true, val_preds, average='macro')

        wandb.log({
            f'fold_{fold_idx+1}_epoch': epoch + 1,
            f'fold_{fold_idx+1}_train_loss': avg_train_loss,
            f'fold_{fold_idx+1}_train_accuracy': train_accuracy,
            f'fold_{fold_idx+1}_train_f1_macro': train_f1,
            f'fold_{fold_idx+1}_val_loss': avg_val_loss,
            f'fold_{fold_idx+1}_val_accuracy': val_accuracy,
            f'fold_{fold_idx+1}_val_f1_macro': val_f1
        })

        print(f"[Fold {fold_idx+1}, Epoch {epoch+1}] "
              f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            torch.save(fold_model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} for fold {fold_idx+1}")
                break

    # ----- LOAD BEST MODEL FOR THIS FOLD & FINAL EVAL -----
    fold_model.load_state_dict(torch.load(best_model_path))
    fold_model.eval()

    val_loss = 0
    val_steps = 0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in DataLoader(encoded_val, batch_size=batch_size):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_t = batch['label'].to(device)

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
    final_val_f1 = f1_score(val_true, val_preds, average='macro')

    fold_val_losses.append(final_val_loss)
    fold_val_accuracies.append(final_val_accuracy)
    fold_val_f1s.append(final_val_f1)

    # ----- PER-CLASS METRICS FOR THIS FOLD -----
    report_dict = classification_report(val_true, val_preds, target_names=class_names, output_dict=True)
    for cname in class_names:
        class_metrics[cname]["precision"].append(report_dict[cname]["precision"])
        class_metrics[cname]["recall"].append(report_dict[cname]["recall"])
        class_metrics[cname]["f1-score"].append(report_dict[cname]["f1-score"])

    print(f"FOLD {fold_idx+1} FINAL VAL METRICS:")
    print(f"Val Loss: {final_val_loss:.4f}")
    print(f"Val Accuracy: {final_val_accuracy:.4f}")
    print(f"Val F1 (macro): {final_val_f1:.4f}")

# -------------- AGGREGATE ACROSS FOLDS --------------
mean_loss = np.mean(fold_val_losses)
std_loss  = np.std(fold_val_losses)
mean_acc  = np.mean(fold_val_accuracies)
std_acc   = np.std(fold_val_accuracies)
mean_f1   = np.mean(fold_val_f1s)
std_f1    = np.std(fold_val_f1s)

print("\n========== 5-FOLD CROSS-VALIDATION AGGREGATE METRICS ==========")
print(f"Val Loss:       {mean_loss:.4f} ± {std_loss:.4f}")
print(f"Val Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Val F1 (macro): {mean_f1:.4f} ± {std_f1:.4f}")

# ---------- PER-CLASS AVERAGE METRICS ----------
print("\n========== PER-CLASS AVERAGE METRICS ACROSS FOLDS ==========")
for cname in class_names:
    prec_mean = np.mean(class_metrics[cname]["precision"])
    prec_std  = np.std(class_metrics[cname]["precision"])
    rec_mean  = np.mean(class_metrics[cname]["recall"])
    rec_std   = np.std(class_metrics[cname]["recall"])
    f1_mean   = np.mean(class_metrics[cname]["f1-score"])
    f1_std    = np.std(class_metrics[cname]["f1-score"])

    print(f"\nClass: {cname}")
    print(f"  Precision: {prec_mean:.3f} ± {prec_std:.3f}")
    print(f"  Recall:    {rec_mean:.3f} ± {rec_std:.3f}")
    print(f"  F1-score:  {f1_mean:.3f} ± {f1_std:.3f}")

wandb.log({
    "cv_mean_val_loss": mean_loss,
    "cv_std_val_loss":  std_loss,
    "cv_mean_val_accuracy": mean_acc,
    "cv_std_val_accuracy":  std_acc,
    "cv_mean_val_f1": mean_f1,
    "cv_std_val_f1":  std_f1
})

for cname in class_names:
    wandb.log({
        f"cv_{cname}_precision_mean": np.mean(class_metrics[cname]["precision"]),
        f"cv_{cname}_precision_std":  np.std(class_metrics[cname]["precision"]),
        f"cv_{cname}_recall_mean":    np.mean(class_metrics[cname]["recall"]),
        f"cv_{cname}_recall_std":     np.std(class_metrics[cname]["recall"]),
        f"cv_{cname}_f1_mean":        np.mean(class_metrics[cname]["f1-score"]),
        f"cv_{cname}_f1_std":         np.std(class_metrics[cname]["f1-score"]),
    })

wandb.finish()