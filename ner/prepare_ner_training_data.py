"""Prepare released clinical NER annotations for the NER training notebook.

The MIMIC-IV-Ext-EntityCoding release stores entity spans in CSV files, while
``ner/ner_model_training.ipynb`` starts from the Label Studio JSON shape used
during annotation. This script joins each note's processed text with its entity
spans and writes that notebook input format without changing the labels.

Inputs default to ``data/mimic-iv-ext-entitycoding/``:
  - ``entity_annotations.csv`` with ``note_id``, ``start``, ``end``, and
    ``entity_label`` columns.
  - ``mimic-iv_notes_subset.csv`` with ``note_id`` and ``text`` columns.

The output defaults to ``results/ner/ner_ac_label_studio_format.json``, which
the notebook's first data-loading cell reads. Each record has this shape:

    {"id": "<note_id>", "text": "<note text>", "label": [
        {"start": 0, "end": 10, "labels": ["disorder"]}
    ]}

Notes follow the order in the notes CSV. Entity spans within each note are
sorted by ``start``, ``end``, and ``entity_label`` so reruns are easy to
compare.

Example:
    python ner/prepare_ner_training_data.py
    python ner/prepare_ner_training_data.py \\
        --annotations data/mimic-iv-ext-entitycoding/entity_annotations.csv \\
        --notes data/mimic-iv-ext-entitycoding/mimic-iv_notes_subset.csv \\
        --output results/ner/ner_ac_label_studio_format.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("prepare_ner_training_data")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RELEASE = REPO_ROOT / "data" / "mimic-iv-ext-entitycoding"
DEFAULT_ANN = DEFAULT_RELEASE / "entity_annotations.csv"
DEFAULT_NOTES = DEFAULT_RELEASE / "mimic-iv_notes_subset.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "ner" / "ner_ac_label_studio_format.json"

REQUIRED_ANNOTATION_COLUMNS = {"note_id", "start", "end", "entity_label"}
REQUIRED_NOTE_COLUMNS = {"note_id", "text"}
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _configure_stdio() -> None:
    """Use UTF-8 streams on Windows terminals when Python exposes reconfigure()."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _validate_columns(frame: pd.DataFrame, required: set[str], source: str | Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required column(s): {', '.join(sorted(missing))}")


def _load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_columns(frame, required_columns, path)
    return frame


def build_label_studio_records(
    annotations: pd.DataFrame,
    notes: pd.DataFrame,
    exclude_labels: set[str] | None = None,
) -> list[dict]:
    """Return records in the Label Studio JSON shape consumed by the notebook."""
    _validate_columns(annotations, REQUIRED_ANNOTATION_COLUMNS, "annotations CSV")
    _validate_columns(notes, REQUIRED_NOTE_COLUMNS, "notes CSV")

    if exclude_labels:
        before = len(annotations)
        annotations = annotations[~annotations["entity_label"].isin(exclude_labels)].copy()
        LOGGER.info(
            "Excluded labels %s: %d rows dropped (%d remaining)",
            sorted(exclude_labels),
            before - len(annotations),
            len(annotations),
        )

    grouped = annotations.groupby("note_id", sort=False)
    records: list[dict] = []
    skipped_missing_text = 0
    skipped_empty_text = 0
    skipped_out_of_bounds = 0

    seen_note_ids: set[str] = set()

    for note_id, text in zip(notes["note_id"], notes["text"]):
        if note_id in seen_note_ids:
            continue
        seen_note_ids.add(note_id)
        if not isinstance(text, str) or not text:
            skipped_empty_text += 1
            continue

        spans = grouped.get_group(note_id) if note_id in grouped.groups else None
        labels: list[dict] = []
        if spans is not None:
            spans = spans.sort_values(["start", "end", "entity_label"], kind="stable")
            text_len = len(text)
            for _, r in spans.iterrows():
                start, end = int(r["start"]), int(r["end"])
                if start < 0 or end > text_len or start >= end:
                    skipped_out_of_bounds += 1
                    continue
                labels.append(
                    {
                        "start": start,
                        "end": end,
                        "labels": [str(r["entity_label"])],
                    }
                )
        records.append({"id": str(note_id), "text": text, "label": labels})

    annotation_only = set(annotations["note_id"].unique()) - seen_note_ids
    if annotation_only:
        skipped_missing_text = len(annotation_only)

    LOGGER.info(
        "Built %d note records (%d entities)",
        len(records),
        sum(len(r["label"]) for r in records),
    )
    if skipped_empty_text:
        LOGGER.warning("Skipped %d note(s) with empty text", skipped_empty_text)
    if skipped_missing_text:
        LOGGER.warning(
            "Skipped %d annotation note_id(s) absent from notes CSV",
            skipped_missing_text,
        )
    if skipped_out_of_bounds:
        LOGGER.warning("Skipped %d annotation row(s) with invalid offsets", skipped_out_of_bounds)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANN,
        help=f"Entity annotations CSV (default: {_relative_to_repo(DEFAULT_ANN)})",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_NOTES,
        help=f"Notes CSV with note_id and text columns (default: {_relative_to_repo(DEFAULT_NOTES)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output Label Studio JSON path (default: {_relative_to_repo(DEFAULT_OUT)})",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="*",
        default=None,
        help="Entity labels to drop. Default: keep all labels.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.annotations.exists():
        LOGGER.error(
            "Annotations file not found: %s. "
            "Place the PhysioNet release files under data/mimic-iv-ext-entitycoding/ "
            "or pass --annotations.",
            args.annotations,
        )
        raise SystemExit(2)
    if not args.notes.exists():
        LOGGER.error(
            "Notes file not found: %s. "
            "Place the PhysioNet release files under data/mimic-iv-ext-entitycoding/ "
            "or pass --notes.",
            args.notes,
        )
        raise SystemExit(2)

    try:
        LOGGER.info("Reading annotations: %s", args.annotations)
        ann = _load_csv(args.annotations, REQUIRED_ANNOTATION_COLUMNS)
        LOGGER.info("Read %d annotation rows", len(ann))
        LOGGER.info("Entity label counts: %s", ann["entity_label"].value_counts().to_dict())

        LOGGER.info("Reading notes: %s", args.notes)
        notes = _load_csv(args.notes, REQUIRED_NOTE_COLUMNS)
        LOGGER.info("Read %d notes", len(notes))

        exclude = set(args.exclude_labels) if args.exclude_labels else None
        records = build_label_studio_records(ann, notes, exclude_labels=exclude)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    LOGGER.info(
        "Wrote %s (%.1f MB)",
        args.output,
        args.output.stat().st_size / 1024 / 1024,
    )


if __name__ == "__main__":
    main()
