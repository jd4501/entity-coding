"""Significance test for the Table 7 metric differences between two models.

This script asks whether the entity-only and full-text PLM-CA models differ
on Table 7 metrics by more than chance, using a paired permutation test on
per-note predictions.

How the test works (one iteration):
  1. For each note (paired by row index after sorting on '_id'), flip a
     coin. When it comes up heads, swap that note's predicted probabilities
     between the two models.
  2. Recompute every metric on the swapped frames and record the (A - B)
     difference.

After ``--num-permutations`` iterations, the per-metric p-value is the
fraction of permuted differences whose magnitude is at least the observed
difference (with the standard +1 / +1 add-one smoothing).

Inputs (one per model):
  Predictions feathers in the schema emitted by the PLM-CA inference path
  (e.g. ``models/{entityonly,fulltext}/predictions_test.feather``). Schema:
  one row per note; columns are ``_id``, ``target`` (ndarray of true codes),
  and one column per ICD code holding the predicted probability.

Outputs:
  Stdout block matching ``results/paper_permutation_test_results.txt``: the
  six classification metrics for each model, the (A - B) differences, and
  the per-metric p-values. Format is preserved so reviewers can diff this
  script's output against the recorded paper run.

Examples:
  # Default: entity-only vs. full-text on the test split, 1,000 permutations.
  # ~6h on a single GPU; matches the recorded paper run.
  python permutation_testing.py

  # Quick smoke test before committing to the full run.
  python permutation_testing.py --num-permutations 5

  # Arbitrary feathers and explicit thresholds.
  python permutation_testing.py \\
      --predictions-a /path/a.feather --threshold-a 0.4 \\
      --predictions-b /path/b.feather --threshold-b 0.45

Run from external/plm_ca/. Use ``--model-*`` and ``--split`` for the released
checkpoint predictions, or pass ``--predictions-*`` paths to compare other
prediction feathers.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from explainable_medical_coding.eval.metrics_permutations import (
    MetricCollection,
    F1Score,
    Precision_K,
    ExactMatchRatio,
    MeanAveragePrecision,
)

LOGGER = logging.getLogger("permutation_testing")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent

# Paper-tuned decision boundaries. Mirror eval_metrics.DEFAULT_THRESHOLDS and
# the inference-script defaults.
DEFAULT_THRESHOLDS = {
    "entityonly": 0.4040403962135315,
    "fulltext": 0.4141414165496826,
}

# Paper-tuned permutation count. The recorded paper run is 1,000 permutations
# at ~6h on a single GPU; output is committed at
# results/paper_permutation_test_results.txt.
DEFAULT_NUM_PERMUTATIONS = 1000
DEFAULT_SEED = 3

EXCLUDED_COLS = frozenset({"_id", "target"})


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def evaluate_predictions(df: pd.DataFrame, model_threshold: float) -> dict:
    """Compute the six classification metrics for one predictions frame."""
    class_cols = [col for col in df.columns if col not in EXCLUDED_COLS]
    if not class_cols:
        raise ValueError("No class probability columns found.")

    class_to_index = {label: idx for idx, label in enumerate(class_cols)}
    num_classes = len(class_cols)

    # Multi-hot encode the per-row 'target' lists.
    targets_array = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, labels in enumerate(df["target"]):
        indices = [class_to_index[label] for label in labels if label in class_to_index]
        targets_array[i, indices] = 1.0
    targets = torch.tensor(targets_array, dtype=torch.float32).to(torch.int64)

    y_probs = torch.tensor(df[class_cols].values, dtype=torch.float32)

    metric_collection = MetricCollection(
        metrics=[
            F1Score(number_of_classes=num_classes, average="micro"),
            F1Score(number_of_classes=num_classes, average="macro"),
            Precision_K(number_of_classes=num_classes, k=8),
            Precision_K(number_of_classes=num_classes, k=15),
            ExactMatchRatio(number_of_classes=num_classes),
            MeanAveragePrecision(number_of_classes=num_classes),
        ],
        threshold=model_threshold,
    )

    metric_collection.update(y_probs, targets)
    return metric_collection.compute(y_probs, targets)


def permutation_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    a_threshold: float,
    b_threshold: float,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    show_progress: bool = True,
) -> dict:
    """Paired permutation test on the per-row predictions of two models.

    Both frames must hold the same set of notes and the same set of class
    columns. Rows are aligned on '_id' when present.
    """
    if len(df_a) != len(df_b):
        raise ValueError(
            "DataFrames must have the same number of samples for permutation testing."
        )

    if "_id" in df_a.columns and "_id" in df_b.columns:
        df_a = df_a.sort_values("_id").reset_index(drop=True)
        df_b = df_b.sort_values("_id").reset_index(drop=True)

    class_cols = [
        col for col in df_a.columns if col not in EXCLUDED_COLS and col in df_b.columns
    ]
    if set(class_cols) != set(df_b.columns) - EXCLUDED_COLS:
        raise ValueError("Class columns in both DataFrames must be identical.")

    num_samples = len(df_a)

    actual_metrics_a = evaluate_predictions(df_a, a_threshold)
    actual_metrics_b = evaluate_predictions(df_b, b_threshold)
    actual_diff = {
        metric: actual_metrics_a[metric] - actual_metrics_b[metric]
        for metric in actual_metrics_a
    }

    perm_diffs: dict[str, list[float]] = defaultdict(list)
    np.random.seed(seed)

    iterator = range(num_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutation Testing")

    for _ in iterator:
        swap_mask = np.random.rand(num_samples) < 0.5

        perm_df_a = df_a.copy()
        perm_df_b = df_b.copy()
        perm_df_a.loc[swap_mask, class_cols] = df_b.loc[swap_mask, class_cols].values
        perm_df_b.loc[swap_mask, class_cols] = df_a.loc[swap_mask, class_cols].values

        metrics_a = evaluate_predictions(perm_df_a, a_threshold)
        metrics_b = evaluate_predictions(perm_df_b, b_threshold)

        for metric in actual_diff:
            if metric in metrics_a and metric in metrics_b:
                diff = metrics_a[metric] - metrics_b[metric]
                if isinstance(diff, torch.Tensor):
                    diff = diff.item()
                perm_diffs[metric].append(diff)

    p_values: dict[str, float] = {}
    for metric, diffs in perm_diffs.items():
        observed_diff = actual_diff.get(metric)
        if observed_diff is None:
            continue
        extreme_count = sum(abs(d) >= abs(observed_diff) for d in diffs)
        p_values[metric] = (extreme_count + 1) / (num_permutations + 1)

    return {
        "actual_metrics_a": actual_metrics_a,
        "actual_metrics_b": actual_metrics_b,
        "actual_diff": actual_diff,
        "perm_diffs": perm_diffs,
        "p_values": p_values,
    }


def _resolve_predictions_path(model: str | None, split: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if model is None:
        raise SystemExit(
            "Either --predictions-* or both --model-* and --split must be set."
        )
    split_filename = {
        "val": "predictions_validation.feather",
        "validation": "predictions_validation.feather",
        "test": "predictions_test.feather",
    }[split]
    return SCRIPT_DIR / "models" / model / split_filename


def _resolve_threshold(model: str | None, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    if model is None:
        raise SystemExit(
            "No threshold given and --model-* not set; cannot infer the paper-tuned default."
        )
    return DEFAULT_THRESHOLDS[model]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired permutation test between two PLM-CA predictions feathers, "
            "for the Table 7 significance check."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-a",
        choices=("entityonly", "fulltext"),
        default="entityonly",
        help="Released-checkpoint shorthand for model A. Default: %(default)s.",
    )
    parser.add_argument(
        "--model-b",
        choices=("entityonly", "fulltext"),
        default="fulltext",
        help="Released-checkpoint shorthand for model B. Default: %(default)s.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "validation", "test"),
        default="test",
        help="Split to use when --predictions-* are not given (default: test).",
    )
    parser.add_argument(
        "--predictions-a",
        type=str,
        default=None,
        help="Path to model A's predictions feather. Overrides --model-a/--split.",
    )
    parser.add_argument(
        "--predictions-b",
        type=str,
        default=None,
        help="Path to model B's predictions feather. Overrides --model-b/--split.",
    )
    parser.add_argument(
        "--threshold-a",
        type=float,
        default=None,
        help=(
            "Decision boundary for model A. Defaults to the paper-tuned value "
            "for --model-a."
        ),
    )
    parser.add_argument(
        "--threshold-b",
        type=float,
        default=None,
        help=(
            "Decision boundary for model B. Defaults to the paper-tuned value "
            "for --model-b."
        ),
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=DEFAULT_NUM_PERMUTATIONS,
        help=(
            "Number of permutation iterations. Default %(default)d matches the "
            "recorded paper run; lower (e.g. 5) for a smoke test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the swap mask (default: %(default)d).",
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

    pred_a = _resolve_predictions_path(args.model_a, args.split, args.predictions_a)
    pred_b = _resolve_predictions_path(args.model_b, args.split, args.predictions_b)
    for label, path in (("A", pred_a), ("B", pred_b)):
        if not path.exists():
            raise SystemExit(
                f"Model {label} predictions feather not found: {path}. "
                f"Run `python data_download.py --models entity-only fulltext` "
                f"from the repo root, or pass --predictions-{label.lower()} explicitly."
            )

    threshold_a = _resolve_threshold(args.model_a, args.threshold_a)
    threshold_b = _resolve_threshold(args.model_b, args.threshold_b)

    LOGGER.info("Model A: %s (threshold %.16f)", pred_a, threshold_a)
    LOGGER.info("Model B: %s (threshold %.16f)", pred_b, threshold_b)
    LOGGER.info("Permutations: %d, seed: %d", args.num_permutations, args.seed)

    df_a = pd.read_feather(pred_a)
    df_b = pd.read_feather(pred_b)

    show_progress = LOGGER.isEnabledFor(logging.INFO)
    results = permutation_test(
        df_a,
        df_b,
        a_threshold=threshold_a,
        b_threshold=threshold_b,
        num_permutations=args.num_permutations,
        seed=args.seed,
        show_progress=show_progress,
    )

    actual_metrics_a = results["actual_metrics_a"]
    actual_metrics_b = results["actual_metrics_b"]
    actual_diff = results["actual_diff"]
    p_values = results["p_values"]

    # Stdout block matches results/paper_permutation_test_results.txt so
    # reviewers can diff this script's output against the recorded paper run.
    print("\n=== Actual Metrics ===")
    print("Model A Metrics:")
    for metric, value in actual_metrics_a.items():
        print(f"{metric}: {value:.4f}")

    print("\nModel B Metrics:")
    for metric, value in actual_metrics_b.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Observed Metric Differences (A - B) ===")
    for metric, diff in actual_diff.items():
        print(f"{metric}: {diff:.4f}")

    print("\n=== Permutation Test P-Values ===")
    for metric, p_val in p_values.items():
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        print(f"{metric}: p-value = {p_val:.4f} ({significance})")


if __name__ == "__main__":
    main()
