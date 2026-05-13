#!/usr/bin/env python3
"""Build entity-only PLM-CA training parquets from extracted entity CSVs.

This script sits between ``ner/extract_entities.py`` and
``external/plm_ca/train_plm_entities.py`` in the entity-only training pipeline.
``extract_entities.py`` emits one row per retained entity. This script groups
those rows by ``note_id``, joins each note's entities with newlines, and
replaces the full-note ``text`` column in a MIMIC-IV ICD-10 parquet shard. The
rest of the parquet schema is left unchanged so the PLM-CA dataset loader can
train on entity-only documents without knowing how they were created.

The default path is the manuscript-faithful data preparation path. Optional
flags can also prepare ablation splits described in manuscript Section 4.4 and
reported in Tables 10 and 11: use ``--remove_tokens``, ``--replace_tokens``, or
``--shuffle`` for the Table 10 entity-token and order ablations. For Table 11
entity-type subset runs, pre-filter the extracted-entities CSV to the desired
entity categories before passing it here.

Example:
    python ner/create_train_input.py \\
        --entities results/ner/mimic-iv-train.csv \\
        --mimic_file external/plm_ca/data/processed/mimiciv_icd10/train.parquet \\
        --output external/plm_ca/data/processed/mimiciv_icd10/entity-only/train.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
import random
import sys

import pandas as pd

LOGGER = logging.getLogger("create_train_input")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

REQUIRED_ENTITY_COLUMNS = {"note_id", "text"}
REQUIRED_PARQUET_COLUMNS = {"note_id", "text"}

RANDOM_SEED = 1
random.seed(RANDOM_SEED)

ENTITY_TOKENS = [
    "<disorder>",
    "<medication>",
    "<procedure>",
    "<health_context>",
    "<abnormal_finding>",
]

PLAIN_TEXT_TOKEN_MAPPING = {
    "<disorder>": "(Disorder)",
    "<medication>": "(Medication)",
    "<procedure>": "(Procedure)",
    "<health_context>": "(Health context)",
    "<abnormal_finding>": "(Abnormal finding)",
}

TOKENS_TO_REMOVE = ENTITY_TOKENS
REPLACEMENT_MAPPING = PLAIN_TEXT_TOKEN_MAPPING


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


def _validate_columns(columns: set[str], required: set[str], source: str | Path) -> None:
    missing = required - columns
    if missing:
        raise ValueError(f"{source} missing required column(s): {', '.join(sorted(missing))}")


def process_text(
    text: str,
    remove_tokens_flag: bool = False,
    replace_tokens_flag: bool = False,
) -> str:
    """Apply the optional Table 10 entity-token ablations to one entity row."""
    if remove_tokens_flag:
        for token in ENTITY_TOKENS:
            text = text.replace(token, "")
    if replace_tokens_flag:
        for token, replacement in PLAIN_TEXT_TOKEN_MAPPING.items():
            text = text.replace(token, replacement)
    return text


def combine_text_by_note_id(
    input_csv: str | Path,
    shuffle: bool = False,
    remove_tokens_flag: bool = False,
    replace_tokens_flag: bool = False,
) -> dict[str, str]:
    """Group entity rows by note and join each note's entities with newlines.

    Entity row order is preserved by default because Section 4.4 tests whether
    local entity and heading order matters. ``--shuffle`` deliberately breaks
    that order for the randomized-order ablation.
    """
    input_csv = Path(input_csv)
    note_texts: defaultdict[str, list[str]] = defaultdict(list)

    with input_csv.open(mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} is empty or has no CSV header")
        _validate_columns(set(reader.fieldnames), REQUIRED_ENTITY_COLUMNS, input_csv)

        for row in reader:
            note_id = row["note_id"]
            text = process_text(
                row["text"],
                remove_tokens_flag=remove_tokens_flag,
                replace_tokens_flag=replace_tokens_flag,
            )
            note_texts[note_id].append(text)

    combined = {}
    for note_id, texts in note_texts.items():
        if shuffle:
            random.shuffle(texts)
        combined[note_id] = "\n".join(texts)

    return combined


def replace_text_in_parquet(
    parquet_file: str | Path,
    combined_texts: dict[str, str],
    output_parquet_file: str | Path,
) -> pd.DataFrame:
    """Replace a full-text parquet's ``text`` column with entity-only text.

    Notes that produced zero entity rows are dropped from the output. Keeping
    their original full text would silently defeat the entity-only pipeline,
    while emitting empty text would feed PLM-CA a degenerate example.
    """
    parquet_file = Path(parquet_file)
    output_parquet_file = Path(output_parquet_file)

    df_parquet = pd.read_parquet(parquet_file)
    _validate_columns(set(df_parquet.columns), REQUIRED_PARQUET_COLUMNS, parquet_file)

    df_csv = pd.DataFrame(list(combined_texts.items()), columns=["note_id", "updated_text"])
    df_merged = df_parquet.merge(df_csv, on="note_id", how="left")

    missing_mask = df_merged["updated_text"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        dropped_ids = df_merged.loc[missing_mask, "note_id"].tolist()
        preview = ", ".join(map(str, dropped_ids[:5]))
        suffix = ", ..." if n_missing > 5 else ""
        LOGGER.warning(
            "Dropping %d note(s) with zero extracted entities: [%s%s]",
            n_missing,
            preview,
            suffix,
        )
        df_merged = df_merged.loc[~missing_mask].copy()

    df_merged["text"] = df_merged["updated_text"]
    df_merged.drop(columns=["updated_text"], inplace=True)

    output_parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(output_parquet_file)
    LOGGER.info("Wrote entity-only parquet: %s", output_parquet_file)
    return df_merged


def remove_note_ids_from_parquet(
    df: pd.DataFrame,
    note_ids_to_remove: list[str],
    output_parquet_file: str | Path,
) -> pd.DataFrame:
    """Drop selected ``note_id`` values and save a filtered parquet copy."""
    output_parquet_file = Path(output_parquet_file)
    df_filtered = df[~df["note_id"].isin(note_ids_to_remove)]
    output_parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_parquet_file)
    LOGGER.info("Wrote filtered parquet: %s", output_parquet_file)
    return df_filtered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--entities",
        type=Path,
        required=True,
        help="Extracted entities CSV from ner/extract_entities.py.",
    )
    parser.add_argument(
        "--mimic_file",
        type=Path,
        required=True,
        help="Full-text MIMIC parquet shard with note_id and text columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path for the entity-only shard.",
    )
    parser.add_argument(
        "--remove_ids",
        nargs="*",
        default=None,
        help="Optional note_id values to remove from the final output.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle entities within each note before joining them.",
    )
    parser.add_argument(
        "--remove_tokens",
        action="store_true",
        help="Remove the five entity type tokens from each entity row.",
    )
    parser.add_argument(
        "--replace_tokens",
        action="store_true",
        help="Replace the five entity type tokens with plain-text labels.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.remove_tokens and args.replace_tokens:
        LOGGER.error("--remove_tokens and --replace_tokens cannot be used together.")
        raise SystemExit(1)

    if not args.entities.exists():
        LOGGER.error(
            "Entities CSV not found: %s. Run ner/extract_entities.py first or pass --entities.",
            _relative_to_repo(args.entities),
        )
        raise SystemExit(2)
    if not args.mimic_file.exists():
        LOGGER.error(
            "MIMIC parquet not found: %s. Run make mimiciv under external/plm_ca or pass --mimic_file.",
            _relative_to_repo(args.mimic_file),
        )
        raise SystemExit(2)

    try:
        combined_texts = combine_text_by_note_id(
            args.entities,
            shuffle=args.shuffle,
            remove_tokens_flag=args.remove_tokens,
            replace_tokens_flag=args.replace_tokens,
        )
        LOGGER.info(
            "Combined extracted entities for %d note_id value(s) from %s",
            len(combined_texts),
            _relative_to_repo(args.entities),
        )

        df_updated = replace_text_in_parquet(args.mimic_file, combined_texts, args.output)

        if args.remove_ids:
            output_filtered = Path(str(args.output).replace(".parquet", "_filtered.parquet"))
            df_final = remove_note_ids_from_parquet(df_updated, args.remove_ids, output_filtered)
            LOGGER.info("Final dataset shape after removal: %s", df_final.shape)
        else:
            LOGGER.info("No note_id values were removed from the entity-only parquet.")
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
