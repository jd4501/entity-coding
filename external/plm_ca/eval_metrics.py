"""Compute ICD coding metrics (Table 7) from a saved predictions feather.

Inputs:
  A predictions feather emitted by the PLM-CA inference path
  (e.g. models/{entityonly,fulltext}/predictions_{validation,test}.feather).
  Schema: one row per note; columns are `_id`, `target` (ndarray of true codes),
  and one column per ICD code holding the predicted probability.

Outputs:
  Stdout table of f1_micro, f1_macro, precision@8, precision@15,
  exact_match_ratio, map, auc_micro, auc_macro. The AUC values use the
  original sklearn-based implementation, matching Table 7. Macro AUC iterates
  `roc_curve` per code, so a full test-set run takes a few minutes.

Examples:
  # Quick val-set check on the released entity-only checkpoint
  python eval_metrics.py --model entityonly --split val

  # Test set on the full-text checkpoint
  python eval_metrics.py --model fulltext --split test

  # Arbitrary feather + custom decision boundary
  python eval_metrics.py --predictions /path/to/predictions_test.feather --threshold 0.4

Run from external/plm_ca/. Defaults assume the directory layout used by the
release `data_download.py` (models under ./models/{entityonly,fulltext}/).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explainable_medical_coding.eval.metrics_permutations import (
    MetricCollection,
    F1Score,
    Precision_K,
    ExactMatchRatio,
    MeanAveragePrecision,
    AUC,
)

LOGGER = logging.getLogger("eval_metrics")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent

# Paper-tuned decision boundaries (also baked into the inference scripts and
# mirrored in permutation_testing.DEFAULT_THRESHOLDS).
DEFAULT_THRESHOLDS = {
    "entityonly": 0.4040403962135315,
    "fulltext": 0.4141414165496826,
}

# Display order. F1 micro listed first to match results/paper_permutation_test_results.txt.
METRIC_ORDER = [
    "f1_micro",
    "f1_macro",
    "precision@8",
    "precision@15",
    "exact_match_ratio",
    "map",
    "auc_micro",
    "auc_macro",
]


def _resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions:
        return Path(args.predictions).expanduser().resolve()
    if not (args.model and args.split):
        raise SystemExit(
            "Specify either --predictions PATH or both --model and --split."
        )
    split_filename = {
        "val": "predictions_validation.feather",
        "validation": "predictions_validation.feather",
        "test": "predictions_test.feather",
    }[args.split]
    return SCRIPT_DIR / "models" / args.model / split_filename


def _resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.model is None:
        raise SystemExit(
            "No --threshold given and --model not set; cannot infer the paper-tuned default."
        )
    return DEFAULT_THRESHOLDS[args.model]


def evaluate_predictions(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Compute the Table 7 metrics on a predictions feather.

    Returns the eight values that appear in manuscript Table 7 (f1_micro,
    f1_macro, precision@8, precision@15, exact_match_ratio, map, auc_micro,
    auc_macro). AUC uses the sklearn-based implementation that the manuscript
    numbers were computed against; macro AUC is the slow step (one roc_curve
    call per code).
    """
    excluded_cols = {"_id", "target"}
    class_cols = [c for c in df.columns if c not in excluded_cols]
    if not class_cols:
        raise ValueError("No class probability columns found in predictions feather.")

    class_to_index = {label: idx for idx, label in enumerate(class_cols)}
    num_classes = len(class_cols)

    targets_array = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, labels in enumerate(df["target"]):
        idxs = [class_to_index[label] for label in labels if label in class_to_index]
        targets_array[i, idxs] = 1.0
    targets = torch.tensor(targets_array, dtype=torch.float32).to(torch.int64)
    y_probs = torch.tensor(df[class_cols].values, dtype=torch.float32)

    metrics: list = [
        F1Score(number_of_classes=num_classes, average="micro"),
        F1Score(number_of_classes=num_classes, average="macro"),
        Precision_K(number_of_classes=num_classes, k=8),
        Precision_K(number_of_classes=num_classes, k=15),
        ExactMatchRatio(number_of_classes=num_classes),
        MeanAveragePrecision(number_of_classes=num_classes),
        AUC(number_of_classes=num_classes, average="micro"),
        AUC(number_of_classes=num_classes, average="macro"),
    ]

    collection = MetricCollection(metrics=metrics, threshold=threshold)
    collection.update(y_probs, targets)
    out = collection.compute(y_probs, targets)
    return {k: float(v) for k, v in out.items()}


def _print_table(label: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    print(f"  {'metric':<20}{'value':>10}")
    for name in METRIC_ORDER:
        if name not in metrics:
            continue
        print(f"  {name:<20}{metrics[name]:>10.4f}")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Table 7 ICD coding metrics from a PLM-CA predictions feather.",
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
        default=None,
        help="Split when using --model shorthand.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision boundary for classification metrics. Defaults to the "
        "paper-tuned value for the chosen --model.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write a machine-readable JSON copy of the metrics.",
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

    LOGGER.info("Predictions: %s", pred_path)
    LOGGER.info("Threshold:   %s", threshold)
    df = pd.read_feather(pred_path)
    LOGGER.info(
        "Rows: %s  Columns: %s (= %s codes + target + _id)",
        f"{len(df):,}",
        f"{df.shape[1]:,}",
        f"{df.shape[1] - 2:,}",
    )

    metrics = evaluate_predictions(df, threshold)

    label = pred_path.name if args.predictions else f"{args.model} {args.split}"
    _print_table(label, metrics)

    if args.json:
        payload = {
            "predictions": str(pred_path),
            "threshold": threshold,
            "rows": int(len(df)),
            "metrics": metrics,
        }
        out_path = Path(args.json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("JSON written to %s", out_path)


if __name__ == "__main__":
    main()
