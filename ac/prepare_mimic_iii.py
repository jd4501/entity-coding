"""Build mimic_assertion_data.csv from bvanaken labels and MIMIC-III notes.

The public label CSVs come from
`github.com/bvanaken/clinical-assertion-data`. They contain 5,000 assertion
labels across discharge summaries, nursing notes, physician notes, and
radiology reports. The note text itself is gated by PhysioNet and comes from
MIMIC-III Clinical Database v1.4 `NOTEEVENTS.csv`.

This script joins those sources by `ROW_ID`, inserts `<entity>` markers around
the labeled span, crops to the surrounding sentence, and writes `text` plus
`assertion` rows for `merge_datasets.py`.

The sentence-boundary helper is a faithful port of the upstream
`convert_to_samples.py`. This project keeps string labels and uses the
`<entity>` marker so the output matches the rest of the AC pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from common import (
        AC_DATA_DIR,
        AC_SOURCES_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_directory,
        require_file,
    )
except ModuleNotFoundError:
    from ac.common import (
        AC_DATA_DIR,
        AC_SOURCES_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_directory,
        require_file,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("prepare_mimic_iii")

LABEL_MAP = {"PRESENT": "present", "ABSENT": "absent", "POSSIBLE": "possible"}

# This order is needed to reproduce the paper-era final split membership.
DEFAULT_LABEL_FILES = (
    "nursing_labels.csv",
    "discharge_summaries_labels.csv",
    "physician_labels.csv",
    "radiology_labels.csv",
)


def index_of_punctuation(s: str, backwards: bool = False) -> int:
    """Return the nearest sentence-ending punctuation index.

    This mirrors bvanaken/clinical-assertion-data, which keeps sentence spans
    aligned with the released label offsets used for the paper splits.
    """
    punctuations = [".", "\n\n", "!", "?"]
    if backwards:
        i_punct = -1
        for punctuation in punctuations:
            i = s.rfind(punctuation)
            if i > i_punct:
                i_punct = i
    else:
        i_punct = 1_000_000
        for punctuation in punctuations:
            i = s.find(punctuation)
            if i != -1 and i < i_punct:
                i_punct = i
    return i_punct


def label_row_to_sentence(start: int, end: int, text: str) -> str:
    """Wrap the span in <entity> markers and crop to the surrounding sentence."""
    text = text[:start] + " <entity> " + text[start:end] + " <entity> " + text[end:]
    sent_start = index_of_punctuation(text[:start], backwards=True) + 1
    sent_end = index_of_punctuation(text[end:], backwards=False) + len(text[:end]) + 1
    return text[sent_start:sent_end].replace("\n", " ").strip()


def load_labels(label_dir: Path) -> pd.DataFrame:
    try:
        require_directory(label_dir, "bvanaken label directory")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Download the four label CSVs from "
            "https://github.com/bvanaken/clinical-assertion-data/tree/main/labels "
            f"into {label_dir} before running this script."
        ) from exc
    frames = []
    for name in DEFAULT_LABEL_FILES:
        try:
            path = require_file(label_dir / name, "bvanaken label CSV")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. Download it from "
                f"https://raw.githubusercontent.com/bvanaken/clinical-assertion-data/main/labels/{name} "
                f"into {label_dir}."
            ) from exc
        frame = pd.read_csv(path)
        require_columns(frame.columns, {"row_id", "start_index", "end_index", "label"}, path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_noteevents(path: Path, row_ids: set[int]) -> dict[int, str]:
    """Stream NOTEEVENTS.csv once and retain only the notes referenced by labels."""
    try:
        require_file(path, "MIMIC-III NOTEEVENTS.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Download NOTEEVENTS.csv.gz from "
            "https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz "
            f"(PhysioNet credentialing required), decompress it, and place the "
            f"CSV at {path}."
        ) from exc
    header = pd.read_csv(path, nrows=0)
    require_columns(header.columns, {"ROW_ID", "TEXT"}, path)

    wanted = set(row_ids)
    found: dict[int, str] = {}
    reader = pd.read_csv(path, usecols=["ROW_ID", "TEXT"], chunksize=50_000)
    for chunk in reader:
        matches = chunk[chunk["ROW_ID"].isin(wanted)]
        for _, row in matches.iterrows():
            found[int(row["ROW_ID"])] = row["TEXT"]
        if len(found) == len(wanted):
            break

    missing = wanted - set(found)
    if missing:
        examples = sorted(missing)[:5]
        LOGGER.warning("%s row_ids were not found in NOTEEVENTS.csv, for example %s", len(missing), examples)
    return found


def build_rows(labels: pd.DataFrame, notes: dict[int, str]):
    for _, ann in labels.iterrows():
        row_id = int(ann["row_id"])
        if row_id not in notes:
            continue
        label = LABEL_MAP.get(str(ann["label"]).strip().upper())
        if label is None:
            continue
        try:
            start = int(ann["start_index"])
            end = int(ann["end_index"])
        except (TypeError, ValueError):
            continue
        sentence = label_row_to_sentence(start, end, notes[row_id])
        yield sentence, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_dir = AC_SOURCES_DIR / "mimic_iii"
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=source_dir / "bvanaken_labels",
        help="Directory containing the four bvanaken *_labels.csv files.",
    )
    parser.add_argument(
        "--noteevents",
        type=Path,
        default=source_dir / "NOTEEVENTS.csv",
        help="Path to the decompressed MIMIC-III NOTEEVENTS.csv file.",
    )
    parser.add_argument("--out", type=Path, default=AC_DATA_DIR / "mimic_assertion_data.csv")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Loading labels from %s", args.labels_dir)
    labels = load_labels(args.labels_dir)
    LOGGER.info("Loaded %s label rows across %s distinct notes", len(labels), labels["row_id"].nunique())

    LOGGER.info("Streaming %s for matching row_ids", args.noteevents)
    notes = load_noteevents(args.noteevents, set(labels["row_id"].astype(int).tolist()))
    LOGGER.info("Matched %s notes", len(notes))

    out_rows = list(build_rows(labels, notes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["text", "assertion"]).to_csv(args.out, index=False)
    LOGGER.info("Wrote %s rows to %s", len(out_rows), args.out)


if __name__ == "__main__":
    main()
