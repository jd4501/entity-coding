"""Recompute the release annotation statistics reported in Table 1.

The MIMIC-IV-Ext-EntityCoding release stores note text, entity spans, and
assertion labels as CSV files. This script reads those release files and prints
the Table 1 summary: entity counts by type, entities per document, words per
entity, and assertion counts by class.

Inputs default to ``data/mimic-iv-ext-entitycoding/``:
  - ``entity_annotations.csv`` with ``entity_id``, ``note_id``, ``start``,
    ``end``, and ``entity_label`` columns.
  - ``assertion_annotations.csv`` with ``entity_id``, ``note_id``, ``start``,
    ``end``, and ``assertion_label`` columns.
  - ``mimic-iv_notes_subset.csv`` with ``note_id`` and ``text`` columns.

Final statistics are printed to stdout. Progress and validation messages use
the standard logger and can be adjusted with ``--log-level``.

Example:
    python ner/compute_release_stats.py
    python ner/compute_release_stats.py \\
        --entities data/mimic-iv-ext-entitycoding/entity_annotations.csv \\
        --assertions data/mimic-iv-ext-entitycoding/assertion_annotations.csv \\
        --notes data/mimic-iv-ext-entitycoding/mimic-iv_notes_subset.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compute_release_stats")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RELEASE = REPO_ROOT / "data" / "mimic-iv-ext-entitycoding"
DEFAULT_ENTITIES = DEFAULT_RELEASE / "entity_annotations.csv"
DEFAULT_ASSERTIONS = DEFAULT_RELEASE / "assertion_annotations.csv"
DEFAULT_NOTES = DEFAULT_RELEASE / "mimic-iv_notes_subset.csv"

REQUIRED_ENTITY_COLUMNS = {"entity_id", "note_id", "start", "end", "entity_label"}
REQUIRED_ASSERTION_COLUMNS = {
    "entity_id",
    "note_id",
    "start",
    "end",
    "assertion_label",
}
REQUIRED_NOTE_COLUMNS = {"note_id", "text"}

ENTITY_LABEL_ORDER = [
    "normal_finding",
    "abnormal_finding",
    "disorder",
    "procedure",
    "health_context",
    "medication",
]
ASSERTION_LABEL_ORDER = [
    "present",
    "absent",
    "possible",
    "hypothetical",
    "not_associated_with_patient",
]


def _configure_stdio() -> None:
    """Use UTF-8 streams on Windows terminals when Python exposes reconfigure()."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _validate_columns(frame: pd.DataFrame, required: set[str], source: str | Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source} missing required column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {list(frame.columns)}"
        )


def _load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_columns(frame, required_columns, path)
    return frame


def _summary(name: str, values: np.ndarray) -> str:
    median = int(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    return f"  {name:30s} median={median} IQR=[{int(q1)}, {int(q3)}]"


def _print_label_counts(
    title: str,
    counts: pd.Series,
    expected_order: list[str],
    total: int,
) -> None:
    print(title)
    for label in expected_order:
        print(f"  {label:30s} {int(counts.get(label, 0)):>7d}")

    unexpected_labels = sorted(set(counts.index) - set(expected_order))
    for label in unexpected_labels:
        LOGGER.warning("Unexpected label in %s: %s", title.lower().rstrip(":"), label)
        print(f"  {label:30s} {int(counts.get(label, 0)):>7d}  (unexpected label)")
    print(f"  {'TOTAL':30s} {total:>7d}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--entities",
        type=Path,
        default=DEFAULT_ENTITIES,
        help=f"Entity annotations CSV (default: {_relative_to_repo(DEFAULT_ENTITIES)})",
    )
    parser.add_argument(
        "--assertions",
        type=Path,
        default=DEFAULT_ASSERTIONS,
        help=f"Assertion annotations CSV (default: {_relative_to_repo(DEFAULT_ASSERTIONS)})",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_NOTES,
        help=f"Notes CSV with note_id and text columns (default: {_relative_to_repo(DEFAULT_NOTES)})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    configure_logging(args.log_level)

    for name, path in [
        ("entities", args.entities),
        ("assertions", args.assertions),
        ("notes", args.notes),
    ]:
        if not path.exists():
            LOGGER.error(
                "%s file not found: %s. See data/README.md for the PhysioNet release layout.",
                name,
                path,
            )
            raise SystemExit(2)

    try:
        LOGGER.info("Reading entities: %s", args.entities)
        entities = _load_csv(args.entities, REQUIRED_ENTITY_COLUMNS)
        LOGGER.info("Read %d entity rows", len(entities))

        LOGGER.info("Reading assertions: %s", args.assertions)
        assertions = _load_csv(args.assertions, REQUIRED_ASSERTION_COLUMNS)
        LOGGER.info("Read %d assertion rows", len(assertions))

        LOGGER.info("Reading notes: %s", args.notes)
        notes = _load_csv(args.notes, REQUIRED_NOTE_COLUMNS)
        LOGGER.info("Read %d notes", len(notes))
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)

    missing_note_ids = set(entities["note_id"].unique()) - set(notes["note_id"].unique())
    if missing_note_ids:
        LOGGER.warning(
            "%d entity note_id(s) are absent from the notes CSV; their span word counts "
            "will be treated as zero.",
            len(missing_note_ids),
        )

    print(f"Notes:       {len(notes):>7d}")
    print(f"Entities:    {len(entities):>7d}")
    print(f"Assertions:  {len(assertions):>7d}")
    print()

    entity_counts = entities["entity_label"].value_counts()
    _print_label_counts("Entities per type:", entity_counts, ENTITY_LABEL_ORDER, len(entities))
    print()

    print("Per-document distribution:")
    per_note = entities.groupby("note_id").size().to_numpy()
    print(_summary("entities per document", per_note))

    raw_span_widths = (entities["end"].astype(int) - entities["start"].astype(int)).to_numpy()
    # Table 1 counts words by slicing each annotated span from the processed
    # note text and applying split(), matching the manuscript's span convention.
    note_text_by_id = dict(zip(notes["note_id"], notes["text"].astype(str)))
    span_word_counts = []
    for _, row in entities.iterrows():
        text = note_text_by_id.get(row["note_id"], "")
        span = text[int(row["start"]): int(row["end"])]
        span_word_counts.append(len(span.split()))
    span_word_counts = np.array(span_word_counts) if span_word_counts else np.array([0])
    print(_summary("words per entity", span_word_counts))
    print(
        f"  (raw span char widths: median={int(np.median(raw_span_widths))}, "
        f"mean={float(np.mean(raw_span_widths)):.1f})"
    )
    print()

    assertion_counts = assertions["assertion_label"].value_counts()
    _print_label_counts(
        "Assertions per class:",
        assertion_counts,
        ASSERTION_LABEL_ORDER,
        len(assertions),
    )


if __name__ == "__main__":
    main()
