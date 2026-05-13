"""Build 2012_train.csv, 2012_test.csv, and i2b2_2012_merged.csv from the i2b2
2012 Temporal Relations release.

The raw tarballs come from the DBMI Data Portal. Put them under
`ac/sources/i2b2_2012/`, or pass `--train-tar` and `--test-tar`.

The 2012 corpus stores assertion status on `<EVENT .../>` elements with
`modality` and `polarity` attributes. This script maps those values to the
label names used by the AC model:

    FACTUAL + POS     maps to present
    FACTUAL + NEG     maps to absent
    POSSIBLE          maps to possible
    CONDITIONAL       maps to hypothetical
    HYPOTHETICAL      maps to hypothetical

The parser mirrors `data/i2b2_2012/get_samples2.py` in the upstream
`bionlplab/assertion_classification_jbi2022` repository
(https://github.com/bionlplab/assertion_classification_jbi2022). You do not
need to clone that repository to run this script; it is referenced only as
the source of the modality and polarity mapping above. `merge_datasets.py`
later drops rows labeled `present` from this source.
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import shutil
from pathlib import Path

import pandas as pd
from lxml import etree

try:
    from common import (
        AC_DATA_DIR,
        AC_SOURCES_DIR,
        add_log_level,
        configure_logging,
        require_file,
        safe_extract_tar,
    )
except ModuleNotFoundError:
    from ac.common import (
        AC_DATA_DIR,
        AC_SOURCES_DIR,
        add_log_level,
        configure_logging,
        require_file,
        safe_extract_tar,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("prepare_i2b2_2012")

LABEL_MAP = {
    ("FACTUAL", "POS"): "present",
    ("FACTUAL", "NEG"): "absent",
    ("POSSIBLE", None): "possible",
    ("CONDITIONAL", None): "hypothetical",
    ("HYPOTHETICAL", None): "hypothetical",
}


def label_for_event(modality: str | None, polarity: str | None) -> str | None:
    if modality == "FACTUAL":
        return LABEL_MAP.get((modality, polarity))
    return LABEL_MAP.get((modality, None))


def process_xml(xml_path: Path):
    """Yield (text, assertion) pairs for every usable EVENT in one XML file."""
    parser = etree.XMLParser(recover=True)
    try:
        tree = etree.parse(str(xml_path), parser)
    except Exception as exc:
        LOGGER.warning("Skipping %s because it could not be parsed: %s", xml_path.name, exc)
        return

    root = tree.getroot()
    text_elements = root.findall(".//TEXT")
    if not text_elements:
        return
    text_content = "".join(text_elements[0].itertext())
    text_content = text_content.replace("\r\n", "\n").replace("\r", "\n")

    for event in root.findall(".//EVENT"):
        modality = event.get("modality")
        polarity = event.get("polarity")
        label = label_for_event(modality, polarity)
        if label is None:
            continue

        try:
            start = int(event.get("start"))
            end = int(event.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end > len(text_content) or start >= end:
            continue

        prev_newline = text_content.rfind("\n", 0, start)
        next_newline = text_content.find("\n", end)
        prev_newline = 0 if prev_newline == -1 else prev_newline + 1
        next_newline = len(text_content) if next_newline == -1 else next_newline
        sentence = text_content[prev_newline:next_newline]

        ent_start = start - prev_newline
        ent_end = end - prev_newline
        modified = (
            sentence[:ent_start]
            + "<entity> "
            + sentence[ent_start:ent_end]
            + " <entity>"
            + sentence[ent_end:]
        )
        yield html.unescape(modified).strip(), label


def process_dir(xml_dir: Path):
    for xml_path in sorted(xml_dir.glob("*.xml")):
        yield from process_xml(xml_path)


def find_xml_dir(root: Path) -> Path:
    """Locate the XML annotation directory in the train or test layout."""
    if any(root.glob("*.xml")):
        return root
    merged = root / "merged_xml"
    if merged.is_dir():
        return merged
    for sub in root.rglob("merged_xml"):
        if sub.is_dir():
            return sub
    raise FileNotFoundError(f"Could not locate *.xml annotations under {root}")


def write_csv(rows, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "assertion"])
        writer.writeheader()
        n = 0
        for text, assertion in rows:
            writer.writerow({"text": text, "assertion": assertion})
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_dir = AC_SOURCES_DIR / "i2b2_2012"
    parser.add_argument(
        "--train-tar",
        type=Path,
        default=source_dir / "2012-07-15.original-annotation.release.tar.gz",
        help="i2b2 2012 train tarball from the DBMI Data Portal.",
    )
    parser.add_argument(
        "--test-tar",
        type=Path,
        default=source_dir / "2012-08-23.test-data.groundtruth.tar.gz",
        help="i2b2 2012 test ground-truth tarball from the DBMI Data Portal.",
    )
    parser.add_argument("--work-dir", type=Path, default=AC_DATA_DIR / "_i2b2_2012_extracted")
    parser.add_argument("--out-dir", type=Path, default=AC_DATA_DIR)
    parser.add_argument("--cleanup", action="store_true", help="Remove work-dir after parsing.")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    for tar in (args.train_tar, args.test_tar):
        try:
            require_file(tar, "Required i2b2 2012 tarball")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. Register at the DBMI Data Portal "
                "(https://portal.dbmi.hms.harvard.edu/projects/n2c2-2012/) and "
                f"place the release tarballs under {tar.parent} before running "
                "this script."
            ) from exc

    LOGGER.info("Extracting %s", args.train_tar.name)
    train_root = safe_extract_tar(args.train_tar, args.work_dir / "train")
    LOGGER.info("Extracting %s", args.test_tar.name)
    test_root = safe_extract_tar(args.test_tar, args.work_dir / "test")

    train_xml = find_xml_dir(train_root)
    test_xml = find_xml_dir(test_root)

    train_csv = args.out_dir / "2012_train.csv"
    test_csv = args.out_dir / "2012_test.csv"
    merged_csv = args.out_dir / "i2b2_2012_merged.csv"

    n_train = write_csv(process_dir(train_xml), train_csv)
    LOGGER.info("Wrote %s rows to %s", n_train, train_csv)
    n_test = write_csv(process_dir(test_xml), test_csv)
    LOGGER.info("Wrote %s rows to %s", n_test, test_csv)

    merged = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    merged.to_csv(merged_csv, index=False)
    LOGGER.info("Wrote %s rows to %s", len(merged), merged_csv)

    if args.cleanup:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        LOGGER.info("Removed %s", args.work_dir)


if __name__ == "__main__":
    main()
