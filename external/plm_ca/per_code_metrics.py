"""Per-code precision / recall / F1 from a PLM-CA predictions feather.

Recomputes the per-code metric tables shipped at
results/per_code_metrics_{entityonly,fulltext}.csv. These tables let readers
compare model performance at the level of individual ICD codes (for example,
to look at how training frequency relates to per-code F1, or which codes the
entity-only model handles better than the full-text baseline).

Inputs:
  * A predictions feather (one row per test note: `_id`, `target` ndarray of
    true codes, and one float column per ICD code holding the predicted
    probability). Default lookup: models/{model}/predictions_test.feather.
  * The PLM-CA training parquet, used only to count how often each code
    appears in training (`train_frequency` column). Default: the entity-only
    train.parquet, since the two committed CSVs share an identical
    `train_frequency` column. Override with `--train-parquet` if needed.
  * The two ICD descriptor CSVs at data/code_descriptions/, for `long_title`.

Outputs:
  CSV with columns code, long_title, precision, recall, f1, tp, fp, fn,
  test_count, train_frequency. One row per code that appears at least once
  in the test ground truth. Codes that appear in test but are NOT in the
  model's classifier head (the model can never predict them) are emitted
  with tp=0, fp=0, fn=test_count so the row still appears alongside its
  train_frequency for downstream analysis.

Note on the rare-code spectrum: upstream Edin et al. dataset prep
(explainable_medical_coding/data/make_mimiciv_icd10.py:29) calls
`remove_rare_codes(..., min_count=10)` on the combined pre-split data,
so every code in the model's label space has at least 10 occurrences
across train+val+test. The `train_frequency` column in the output CSV
counts only the train split, so it can drop below 10 (about 15% of codes
fall in that range) when most of a code's occurrences happen to land in
val/test. These rare-in-train codes were still part of training and do
produce successful predictions; treat low-train_frequency rows as
expected, not as a bug or a leakage signal.

Examples:
  python per_code_metrics.py --model entityonly
  python per_code_metrics.py --model fulltext
  python per_code_metrics.py --predictions PATH --threshold 0.4 --output OUT.csv

Run from external/plm_ca/. Defaults assume the directory layout used by
data_download.py (models under ./models/{entityonly,fulltext}/).
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_metrics import (
    DEFAULT_THRESHOLDS,
    LOG_LEVELS,
    _resolve_predictions_path,
    _resolve_threshold,
    configure_logging,
)

LOGGER = logging.getLogger("per_code_metrics")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_TRAIN_PARQUET = (
    SCRIPT_DIR
    / "data"
    / "processed"
    / "mimiciv_icd10"
    / "entity-only"
    / "train.parquet"
)
DIAGNOSES_FILE = REPO_ROOT / "data" / "code_descriptions" / "d_icd_diagnoses.csv"
PROCEDURES_FILE = REPO_ROOT / "data" / "code_descriptions" / "d_icd_procedures.csv"

DEFAULT_OUTPUTS = {
    "entityonly": REPO_ROOT / "results" / "per_code_metrics_entityonly.csv",
    "fulltext": REPO_ROOT / "results" / "per_code_metrics_fulltext.csv",
}

OUTPUT_COLUMNS = [
    "code",
    "long_title",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "test_count",
    "train_frequency",
]

# Same epsilon as eval_per_code in explainable_medical_coding/eval/metrics.py;
# keeps divide-by-zero rows at exactly 0 instead of NaN.
EPS = 1e-8


def _normalise_code(code: str) -> str:
    """Strip dots and uppercase so dotted predictions match dotless descriptors."""
    return code.replace(".", "").upper()


def load_long_titles() -> dict[str, str]:
    """Build a {normalised ICD-10 code: long_title} map from the descriptor CSVs.

    Both d_icd_diagnoses.csv and d_icd_procedures.csv mix ICD-9 and ICD-10
    rows; only icd_version == 10 is kept. Diagnosis codes appear dotless in
    these files but dotted in the prediction feather, so both sides are
    normalised through `_normalise_code` before lookup.
    """
    titles: dict[str, str] = {}
    for path in (DIAGNOSES_FILE, PROCEDURES_FILE):
        if not path.exists():
            LOGGER.warning("ICD descriptor not found: %s", path)
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("icd_version") != "10":
                    continue
                titles[_normalise_code(row["icd_code"])] = row["long_title"]
    return titles


def compute_train_frequency(parquet_path: Path) -> Counter:
    """Count how many times each ICD code appears across training notes."""
    df = pd.read_parquet(parquet_path)
    counter: Counter = Counter()
    for column in ("diagnosis_codes", "procedure_codes"):
        if column not in df.columns:
            continue
        for cell in df[column]:
            if isinstance(cell, np.ndarray):
                counter.update(cell.tolist())
    return counter


def _multi_hot_targets(
    target_lists, class_to_index: dict[str, int]
) -> tuple[np.ndarray, Counter]:
    """Build the (N, C) multi-hot target matrix and tally raw test occurrences.

    The Counter tallies every code in the test target column, including codes
    not in the model's label space; the matrix only marks codes the model
    can predict. The two are reconciled in `compute_per_code_metrics`.
    """
    num_classes = len(class_to_index)
    targets = np.zeros((len(target_lists), num_classes), dtype=np.float32)
    test_counts: Counter = Counter()
    for i, labels in enumerate(target_lists):
        for label in labels:
            test_counts[label] += 1
            idx = class_to_index.get(label)
            if idx is not None:
                targets[i, idx] = 1.0
    return targets, test_counts


def compute_per_code_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-code TP/FP/FN/precision/recall/F1 from a predictions feather.

    Codes in the model's classifier head get TP/FP/FN from thresholded
    predictions. Codes that appear in test ground truth but not in the head
    get tp=fp=0, fn=test_count: the model has no output column for them, so
    every test occurrence is a missed prediction. The output is filtered to
    codes with test_count > 0 (otherwise precision/recall/F1 are undefined).
    """
    excluded = {"_id", "target"}
    class_cols = [c for c in df.columns if c not in excluded]
    if not class_cols:
        raise ValueError("No class probability columns in predictions feather.")

    class_to_index = {label: idx for idx, label in enumerate(class_cols)}
    targets_np, test_counts = _multi_hot_targets(df["target"], class_to_index)

    targets = torch.tensor(targets_np, dtype=torch.long)
    y_probs = torch.tensor(df[class_cols].values, dtype=torch.float32)
    predictions = (y_probs > threshold).long()

    tp = (predictions * targets).sum(axis=0).numpy()
    fp = (predictions * (1 - targets)).sum(axis=0).numpy()
    fn = ((1 - predictions) * targets).sum(axis=0).numpy()

    rows: list[dict] = []
    for code, idx in class_to_index.items():
        count = test_counts.get(code, 0)
        if count == 0:
            continue
        rows.append({
            "code": code,
            "tp": int(tp[idx]),
            "fp": int(fp[idx]),
            "fn": int(fn[idx]),
            "test_count": int(count),
        })

    in_label_space = set(class_to_index)
    for code, count in test_counts.items():
        if code in in_label_space:
            continue
        rows.append({
            "code": code,
            "tp": 0,
            "fp": 0,
            "fn": int(count),
            "test_count": int(count),
        })

    out = pd.DataFrame(rows)
    out["precision"] = out["tp"] / (out["tp"] + out["fp"] + EPS)
    out["recall"] = out["tp"] / (out["tp"] + out["fn"] + EPS)
    out["f1"] = (
        2 * out["precision"] * out["recall"]
        / (out["precision"] + out["recall"] + EPS)
    )
    return out


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    if args.model is None:
        raise SystemExit(
            "No --output given and --model not set; cannot infer default output path."
        )
    return DEFAULT_OUTPUTS[args.model]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-code precision/recall/F1 from a PLM-CA predictions "
            "feather. Defaults reproduce the committed "
            "results/per_code_metrics_{entityonly,fulltext}.csv files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to a predictions feather. Overrides --model/--split.",
    )
    parser.add_argument(
        "--model",
        choices=("entityonly", "fulltext"),
        default=None,
        help="Released-checkpoint shorthand. Looks under ./models/<MODEL>/.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "validation", "test"),
        default="test",
        help="Split when using --model shorthand (default: test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Decision boundary for binarising predictions. Defaults to the "
            "paper-tuned value for the chosen --model "
            f"(entityonly={DEFAULT_THRESHOLDS['entityonly']}, "
            f"fulltext={DEFAULT_THRESHOLDS['fulltext']})."
        ),
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default=None,
        help=(
            "Training parquet to count train_frequency from. Default: "
            "data/processed/mimiciv_icd10/entity-only/train.parquet."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV path. Default when --model is set: "
            "results/per_code_metrics_<MODEL>.csv "
            "(overwrites the committed paper artefact)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    pred_path = _resolve_predictions_path(args)
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")
    threshold = _resolve_threshold(args)
    train_parquet = (
        Path(args.train_parquet).expanduser().resolve()
        if args.train_parquet
        else DEFAULT_TRAIN_PARQUET
    )
    if not train_parquet.exists():
        raise SystemExit(f"Train parquet not found: {train_parquet}")
    output_path = _resolve_output_path(args)

    LOGGER.info("Predictions:   %s", pred_path)
    LOGGER.info("Threshold:     %s", threshold)
    LOGGER.info("Train parquet: %s", train_parquet)
    LOGGER.info("Output:        %s", output_path)

    df = pd.read_feather(pred_path)
    LOGGER.info(
        "Predictions rows: %s  Codes in label space: %s",
        f"{len(df):,}",
        f"{df.shape[1] - 2:,}",
    )

    metrics = compute_per_code_metrics(df, threshold)
    LOGGER.info("Codes with test_count > 0: %s", f"{len(metrics):,}")

    LOGGER.info("Counting train-set frequencies from %s", train_parquet.name)
    train_freq = compute_train_frequency(train_parquet)
    # Codes absent from train get NaN (not 0) so downstream analyses can
    # distinguish "not seen during training" from "seen exactly zero times".
    metrics["train_frequency"] = metrics["code"].map(
        lambda c: float(train_freq[c]) if c in train_freq else np.nan
    )
    missing_train = int(metrics["train_frequency"].isna().sum())
    if missing_train:
        LOGGER.info(
            "%d codes appear in test but not in the training parquet (train_frequency=NaN).",
            missing_train,
        )

    titles = load_long_titles()
    metrics["long_title"] = metrics["code"].map(
        lambda c: titles.get(_normalise_code(c), "")
    )
    missing_titles = int((metrics["long_title"] == "").sum())
    if missing_titles:
        LOGGER.warning(
            "%d codes have no long_title in d_icd_{diagnoses,procedures}.csv.",
            missing_titles,
        )

    metrics = metrics.sort_values("code", kind="stable").reset_index(drop=True)
    metrics = metrics[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", f"{len(metrics):,}", output_path)


if __name__ == "__main__":
    main()
