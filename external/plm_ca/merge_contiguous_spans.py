"""Merge per-word evidence into contiguous character-level spans.

Second stage of the full-text evidence pipeline. Reads the per-word output of
infer_with_explanations_fulltext.py and groups consecutive words into spans,
producing one row per (note_id, predicted_code) pair with parallel lists of
merged-span texts, spans, and mean attributions. Merged spans are the
meaningful evidence unit for reporting; single words alone rarely convey a
predicted code's rationale.

Usage (from external/plm_ca/, with the entitycoding conda env active):

    python merge_contiguous_spans.py <input_basename> [output_basename] [attr_threshold]

``<input_basename>`` may include a path prefix. Matching the canonical layout
used by infer_with_explanations_fulltext.py, invocations read the fft_ outputs
from the repo root via
``../../fft_inferred_notes_with_evidence_<split>_DConly`` and write merged
outputs back to the repo root.

If ``<input_basename>.parquet`` exists it is preferred (list columns survive
round-trip). Otherwise ``<input_basename>.csv`` is read and the list columns
are parsed with ``ast.literal_eval``.

End-to-end full-text evidence flow:

  1. Run infer_with_explanations_fulltext.py over every MDACE-DC split (val,
     train, test) at a permissive ``1e-6`` so every per-word attribution is
     retained. See that script's docstring (and Stage 1 of
     code_evidence/code_evidence_eval.ipynb) for the discharge-summary
     filtering applied first.
  2. Tune ``attr_threshold`` on the val per-word output in Stage 2 of the
     notebook by sweeping for the value that maximises partial-match F2
     against MDACE ground-truth evidence spans. Reference point: a best
     threshold of ``0.0013877551020408164`` gave a partial F2 of ``0.4079``
     on val; the operating point where per-word precision/recall were well
     balanced for this task.
  3. Run this script on the train and test per-word outputs, passing the
     val-derived threshold as ``attr_threshold``. Stage 4 of the notebook
     then evaluates on the concatenated (train + test) merged output.

Words with ``abs(attr) < attr_threshold`` are dropped *before* merging. If
that removes every word for a (note_id, predicted_code) pair, the single word
with the largest absolute attribution is kept (the same >=1-evidence
guarantee infer_with_explanations_fulltext.py applies at extraction time).
``attr_threshold`` is optional: when omitted, all input words are kept and
merged as-is (useful for inspecting raw output).

Conventions worth knowing:
  * Adjacency rule: two words are contiguous if the next word's start offset
    equals the previous word's end offset, or that end offset + 1. This
    collapses consecutive tokens separated by at most one whitespace
    character into a single span.
  * Span attribution is the *mean* of the constituent word attributions.
  * Rows are sorted by (note_id ascending, predicted_code_probability
    descending) before write, matching the order
    infer_with_explanations.py emits.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOGGER = logging.getLogger("merge_contiguous_spans")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

REQUIRED_INPUT_COLUMNS = (
    "note_id",
    "predicted_code",
    "predicted_code_probability",
    "evidence_texts",
    "evidence_spans",
    "evidence_attributions",
)


def find_longest_contiguous_spans(words: list[dict]) -> list[dict]:
    """Group words into maximal contiguous blocks by character offsets.

    Words are sorted by start offset, then walked left-to-right. A new word
    extends the current block when its start offset equals the previous
    word's end offset or that end + 1 (i.e., the words are adjacent or
    separated by a single whitespace character in the source text).
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: w["start"])

    results: list[dict] = []
    current_block = [words[0]]
    for i in range(1, len(words)):
        prev_word = words[i - 1]
        current_word = words[i]
        if (
            current_word["start"] == prev_word["end"] + 1
            or current_word["start"] == prev_word["end"]
        ):
            current_block.append(current_word)
        else:
            results.append(_create_span_record(current_block))
            current_block = [current_word]
    if current_block:
        results.append(_create_span_record(current_block))
    return results


def _create_span_record(block: list[dict]) -> dict:
    span_text = " ".join(w["word"] for w in block)
    span_start = block[0]["start"]
    span_end = block[-1]["end"]
    avg_attr = sum(w["attr"] for w in block) / len(block)
    return {
        "span_text": span_text,
        "span_start": span_start,
        "span_end": span_end,
        "span_attr": avg_attr,
    }


def _filter_by_threshold(words: list[dict], attr_threshold: float) -> list[dict]:
    """Drop words below threshold; keep the single strongest if all fall below.

    Mirrors the >=1-evidence guarantee in infer_with_explanations_fulltext.py.
    """
    filtered = [w for w in words if abs(w["attr"]) >= attr_threshold]
    if not filtered and words:
        filtered = [max(words, key=lambda w: abs(w["attr"]))]
    return filtered


def _strip_known_suffix(name: str) -> str:
    """Strip a single trailing .csv or .parquet suffix if present."""
    for suffix in (".csv", ".parquet"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load(input_basename: str) -> pd.DataFrame:
    """Load the per-word evidence file, preferring parquet over csv.

    List columns survive a parquet round-trip but become string literals in
    CSV; ``ast.literal_eval`` is used to revive them. Missing both formats is
    a fail-early condition.
    """
    parquet_path = Path(f"{input_basename}.parquet")
    csv_path = Path(f"{input_basename}.csv")

    if parquet_path.exists():
        LOGGER.info("Reading per-word evidence from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        LOGGER.info("Reading per-word evidence from %s", csv_path)
        df = pd.read_csv(csv_path)
        for col in ("evidence_texts", "evidence_spans", "evidence_attributions"):
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    else:
        LOGGER.error(
            "Could not find input at %s or %s. Run "
            "infer_with_explanations_fulltext.py first.",
            parquet_path,
            csv_path,
        )
        sys.exit(1)

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.error("Input missing required columns: %s", missing)
        LOGGER.error("Found columns: %s", list(df.columns))
        sys.exit(1)
    return df


def post_process_longest_spans(
    input_basename: str,
    output_basename: str,
    attr_threshold: float | None = None,
) -> pd.DataFrame:
    """Merge per-word evidence into contiguous spans and write CSV + parquet.

    Returns the merged DataFrame. Output rows are sorted by (note_id ascending,
    predicted_code_probability descending) for stable diffs across reruns.
    """
    df = _load(input_basename)

    all_rows: list[dict] = []
    for _, row in df.iterrows():
        words_data = [
            {"word": w, "start": int(s), "end": int(e), "attr": float(a)}
            for w, (s, e), a in zip(
                row["evidence_texts"],
                row["evidence_spans"],
                row["evidence_attributions"],
            )
        ]
        if attr_threshold is not None:
            words_data = _filter_by_threshold(words_data, attr_threshold)
        contiguous_spans = find_longest_contiguous_spans(words_data)

        all_rows.append(
            {
                "note_id": row["note_id"],
                "predicted_code": row["predicted_code"],
                "predicted_code_probability": row["predicted_code_probability"],
                "evidence_texts": [sp["span_text"] for sp in contiguous_spans],
                "evidence_spans": [
                    (sp["span_start"], sp["span_end"]) for sp in contiguous_spans
                ],
                "evidence_attributions": [sp["span_attr"] for sp in contiguous_spans],
            }
        )

    merged = pd.DataFrame(all_rows).sort_values(
        by=["note_id", "predicted_code_probability"],
        ascending=[True, False],
    )

    output_csv = Path(f"{output_basename}.csv")
    output_parquet = Path(f"{output_basename}.parquet")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    merged.to_parquet(output_parquet, index=False)
    LOGGER.info("Merged spans written to %s", output_csv)
    LOGGER.info("Merged spans written to %s", output_parquet)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-word evidence from infer_with_explanations_fulltext.py "
            "into contiguous character-level spans."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (from external/plm_ca/, applying a val-tuned threshold):\n"
            "  python merge_contiguous_spans.py \\\n"
            "      ../../fft_inferred_notes_with_evidence_mdace_train_DConly \\\n"
            "      ../../fft_inferred_notes_with_evidence_mdace_train_DConly_merged \\\n"
            "      0.0013877551020408164"
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Per-word output basename from infer_with_explanations_fulltext.py "
            "(no suffix; .parquet preferred over .csv if both exist)."
        ),
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=(
            "Output prefix for the merged file (no suffix; both .csv and "
            ".parquet are written). Default: <input_file>_merged."
        ),
    )
    parser.add_argument(
        "attr_threshold",
        nargs="?",
        type=float,
        default=None,
        help=(
            "Optional per-word attribution cutoff applied before merging. "
            "Words with abs(attr) below this are dropped, with a fallback to "
            "the single highest-attr word per code. Omit to merge all input "
            "words as-is."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    input_basename = _strip_known_suffix(args.input_file)
    if args.output_file is not None:
        output_basename = _strip_known_suffix(args.output_file)
    else:
        output_basename = f"{input_basename}_merged"

    if args.attr_threshold is not None:
        LOGGER.info("Applying per-word attribution threshold %g", args.attr_threshold)
    else:
        LOGGER.info(
            "No attribution threshold supplied; merging all input words as-is."
        )

    post_process_longest_spans(input_basename, output_basename, args.attr_threshold)


if __name__ == "__main__":
    main()
