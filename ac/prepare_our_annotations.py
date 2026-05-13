"""Build our_new_assertions.csv from MIMIC-IV-Ext-EntityCoding.

Download the PhysioNet release into `data/mimic-iv-ext-entitycoding/`. For AC
training this script reads `assertion_sentences.csv`, wraps each target span
in `<entity>` markers, and writes `text` plus `assertion` rows for
`merge_datasets.py`.

The release label `not_associated_with_patient` is mapped to the i2b2-style
label `associated_with_someone_else` so all AC training sources share one
label vocabulary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from common import (
        AC_DATA_DIR,
        ENTITYCODING_RELEASE_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_file,
    )
except ModuleNotFoundError:
    from ac.common import (
        AC_DATA_DIR,
        ENTITYCODING_RELEASE_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_file,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("prepare_our_annotations")

REQUIRED_COLUMNS = {
    "entity_id",
    "note_id",
    "text",
    "entity_text",
    "entity_start",
    "entity_end",
    "assertion_label",
}

LABEL_ALIASES = {"not_associated_with_patient": "associated_with_someone_else"}


def wrap_entity(text: str, start: int, end: int) -> str:
    return text[:start] + "<entity> " + text[start:end] + " <entity>" + text[end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sentences",
        type=Path,
        default=ENTITYCODING_RELEASE_DIR / "assertion_sentences.csv",
        help="PhysioNet-format assertion_sentences.csv from data/mimic-iv-ext-entitycoding/.",
    )
    parser.add_argument("--out", type=Path, default=AC_DATA_DIR / "our_new_assertions.csv")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        require_file(args.sentences, "MIMIC-IV-Ext-EntityCoding assertion_sentences.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Download the PhysioNet release into "
            f"{ENTITYCODING_RELEASE_DIR} before running this script."
        ) from exc

    df = pd.read_csv(args.sentences)
    require_columns(df.columns, REQUIRED_COLUMNS, args.sentences)

    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        start = int(row["entity_start"])
        end = int(row["entity_end"])
        if pd.isna(text) or start < 0 or start >= len(text):
            continue

        # Some released rows have a target span that ends at the truncated
        # sentence boundary. Clamping preserves the visible span in the CSV.
        end = min(end, len(text))
        if end <= start:
            continue

        assertion = LABEL_ALIASES.get(row["assertion_label"], row["assertion_label"])
        rows.append({"text": wrap_entity(text, start, end), "assertion": assertion})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["text", "assertion"]).to_csv(args.out, index=False)
    LOGGER.info("Wrote %s rows to %s", len(rows), args.out)
    if rows:
        counts = pd.DataFrame(rows)["assertion"].value_counts().to_dict()
        LOGGER.info("Label counts: %s", counts)


if __name__ == "__main__":
    main()
