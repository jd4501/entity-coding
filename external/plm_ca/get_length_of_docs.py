"""Compute the document-length statistics reported in Table 5 of the paper.

This script measures how much shorter the entity-only documents are than the
raw discharge summaries they are distilled from. It expects two side-by-side
parquet trees holding the PLM-CA training inputs:

  - ``entity-only/{train,val,test}.parquet``: the consolidated entity
    documents produced by ``ner/create_train_input.py``. Each note's text is
    its detected entities joined with newlines, with the five entity tags
    (``<disorder>``, ``<medication>``, ``<procedure>``, ``<health_context>``,
    ``<abnormal_finding>``) retained inline.
  - ``fulltext/{train,val,test}.parquet``: the upstream PLM-CA full-text
    parquets produced by ``make mimiciv`` (no entity tags).

For each tree it concatenates train+val+test, strips the entity tags, splits
on whitespace, and prints the word-count median, IQR, and total. Those are
the numbers that appear in Table 5.

Why strip the entity tags first? They are markup we inject into the
entity-only documents so PLM-CA can recognise entity boundaries and types;
the source discharge summaries do not contain them. The point of Table 5
is to report how much of the *original clinical content* survives entity
extraction (i.e. content retention). Stripping them puts both halves of the comparison
(entity-only and full-text) in the same units, words of source clinical
text, so the reduction figure reflects what was kept from the note.

Inputs are not committed to this repo because they derive from credentialed
MIMIC downloads. See docs/reproduce.md#table-5-recipe for the prerequisite
``make mimiciv`` and entity-extraction steps that produce them. Defaults assume the directory layout used inside ``external/plm_ca/``
(``data/processed/mimiciv_icd10/``); pass ``--data-root`` to point elsewhere.

Example:
    cd external/plm_ca
    python get_length_of_docs.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("get_length_of_docs")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data" / "processed" / "mimiciv_icd10"

# The five entity tags used across NER, AC filtering, and PLM-CA training.
# Stripped before whitespace tokenisation so the word counts reflect original
# note content (what survived entity extraction) rather than the synthetic
# markup we add for the model. See module docstring for the full rationale.
ENTITY_TOKENS = (
    "<disorder>",
    "<medication>",
    "<procedure>",
    "<health_context>",
    "<abnormal_finding>",
)

SPLITS = ("train", "val", "test")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def strip_entity_tokens(text: str) -> str:
    """Remove the five entity tags so word counts measure source clinical content.

    The tags are markup added to the entity-only documents for PLM-CA's
    benefit and are not part of the source note. Removing them before
    counting keeps the entity-only and full-text figures in the same units
    (words of source text), so Table 5 reads as a content-retention number.
    """
    for token in ENTITY_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def load_split_union(folder: Path) -> pd.DataFrame:
    """Concatenate train/val/test parquets under ``folder`` into one frame."""
    frames = []
    for split in SPLITS:
        path = folder / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing parquet shard: {path}. "
                f"Expected {SPLITS} under {folder}."
            )
        frame = pd.read_parquet(path)
        if "text" not in frame.columns:
            raise ValueError(
                f"{path} has no 'text' column (found: {list(frame.columns)})."
            )
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True)


def summarise_lengths(folder: Path, label: str) -> pd.DataFrame:
    """Print median/IQR/total word counts for one parquet tree."""
    LOGGER.info("Loading %s parquets from %s", label, folder)
    df = load_split_union(folder)

    text_stripped = df["text"].astype(str).map(strip_entity_tokens)
    lengths = text_stripped.str.split().map(len)

    q1 = lengths.quantile(0.25)
    median = lengths.median()
    q3 = lengths.quantile(0.75)
    iqr = q3 - q1
    total = lengths.sum()

    # Stdout lines are the script's intentional output interface; keep them as
    # plain prints so callers can grep the result without filtering log noise.
    print(f"{label.upper()} DATA:")
    print(f"Median doc length: {median} with IQR {iqr} ({q1} - {q3})")
    print(f"Total words: {total}\n")

    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the entity-only and full-text document-length statistics "
            "reported in Table 5 of the paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd external/plm_ca\n"
            "  python get_length_of_docs.py"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing 'entity-only/' and 'fulltext/' subfolders, "
            "each with train.parquet/val.parquet/test.parquet. Default: "
            "%(default)s."
        ),
    )
    parser.add_argument(
        "--entity-only-only",
        action="store_true",
        help=(
            "Only summarise the entity-only tree. Useful when the full-text "
            "parquets are not present locally (the entity-only tree is the "
            "more commonly available half)."
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

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(
            f"--data-root {data_root} does not exist. Run `make mimiciv` "
            f"and the entity-extraction pipeline first."
        )

    summarise_lengths(data_root / "entity-only", "entity-only")
    if not args.entity_only_only:
        summarise_lengths(data_root / "fulltext", "fulltext")


if __name__ == "__main__":
    main()
