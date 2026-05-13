"""Build 2010_train.csv and 2010_test.csv from the i2b2 2010 tarballs.

The raw files come from the DBMI Data Portal. Put the three release tarballs
under `ac/sources/i2b2_2010/`, or pass explicit paths with `--train-tar`,
`--test-data-tar`, and `--test-ref-tar`.

The assertion files (*.ast) have one record per line:
    c="<concept text>" <start_line>:<start_word> <end_line>:<end_word>||t="..."||a="<assertion>"

Each output row has `text` and `assertion`. The text is the source sentence
with the concept wrapped in `<entity> ... <entity>` markers. Labels are kept
verbatim; `merge_datasets.py` drops the `conditional` rows later.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

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
LOGGER = logging.getLogger("prepare_i2b2_2010")


def parse_ast_line(line: str) -> tuple[str, int, int, int, int, str] | None:
    """Parse one .ast record and return span coordinates plus the label."""
    line = line.strip()
    if not line:
        return None
    parts = line.split("||")
    if len(parts) < 3:
        return None
    c_part, a_part = parts[0], parts[-1]
    try:
        c_text = c_part.split('" ')[0][3:]
        positions = c_part.split()[-2:]
        start_line, start_word = map(int, positions[0].split(":"))
        end_line, end_word = map(int, positions[1].split(":"))
        assertion = a_part.split("=", 1)[1].strip().strip('"')
    except (IndexError, ValueError):
        return None
    return c_text, start_line, start_word, end_line, end_word, assertion


def build_sample(
    txt_lines: list[str],
    c_text: str,
    start_line: int,
    start_word: int,
    end_line: int,
    end_word: int,
) -> str:
    """Return the source sentence with <entity> markers around c_text."""
    s_idx = start_line - 1
    e_idx = end_line - 1
    source_line = txt_lines[s_idx].strip()
    line_words = source_line.split()
    if s_idx == e_idx:
        line_words = (
            line_words[:start_word]
            + ["<entity> " + c_text + " <entity>"]
            + line_words[end_word + 1 :]
        )
    else:
        line_words = line_words[:start_word] + ["<entity> " + c_text + " <entity>"]
    return " ".join(line_words)


def process_split(ast_txt_pairs, output_csv: Path) -> int:
    """Write one CSV from an iterable of (.ast path, .txt path) pairs."""
    samples = []
    for ast_path, txt_path in ast_txt_pairs:
        if not txt_path.exists():
            LOGGER.warning(
                "Skipping %s because the matching text file is missing: %s",
                ast_path.name,
                txt_path,
            )
            continue
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            txt_lines = f.readlines()
        with open(ast_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_ast_line(line)
                if parsed is None:
                    continue
                c_text, sl, sw, el, ew, assertion = parsed
                if sl - 1 >= len(txt_lines):
                    continue
                text = build_sample(txt_lines, c_text, sl, sw, el, ew)
                samples.append([text, assertion])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "assertion"])
        writer.writerows(samples)
    return len(samples)


def train_pairs(train_root: Path):
    """Yield train .ast and .txt pairs from the i2b2 2010 train layout."""
    for institution in ("beth", "partners"):
        ast_dir = train_root / institution / "ast"
        txt_dir = train_root / institution / "txt"
        if not ast_dir.is_dir() or not txt_dir.is_dir():
            continue
        for ast in sorted(ast_dir.glob("*.ast")):
            yield ast, txt_dir / (ast.stem + ".txt")


def test_pairs(test_txt_root: Path, test_ast_root: Path):
    """Yield test .ast and .txt pairs by matching file stems."""
    ast_dir = test_ast_root / "ast" if (test_ast_root / "ast").is_dir() else test_ast_root
    for ast in sorted(ast_dir.glob("*.ast")):
        yield ast, test_txt_root / (ast.stem + ".txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_dir = AC_SOURCES_DIR / "i2b2_2010"
    parser.add_argument(
        "--train-tar",
        type=Path,
        default=source_dir / "concept_assertion_relation_training_data.tar.gz",
        help="i2b2 2010 training tarball from the DBMI Data Portal.",
    )
    parser.add_argument(
        "--test-data-tar",
        type=Path,
        default=source_dir / "test_data.tar.gz",
        help="i2b2 2010 test text tarball from the DBMI Data Portal.",
    )
    parser.add_argument(
        "--test-ref-tar",
        type=Path,
        default=source_dir / "reference_standard_for_test_data.tar.gz",
        help="i2b2 2010 test reference-standard tarball from the DBMI Data Portal.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=AC_DATA_DIR / "_i2b2_2010_extracted",
        help="Directory where tarballs are unpacked. Removed when --cleanup is set.",
    )
    parser.add_argument("--out-dir", type=Path, default=AC_DATA_DIR)
    parser.add_argument("--cleanup", action="store_true", help="Remove work-dir after parsing.")
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    for tar in (args.train_tar, args.test_data_tar, args.test_ref_tar):
        try:
            require_file(tar, "Required i2b2 2010 tarball")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. Register at the DBMI Data Portal "
                "(https://portal.dbmi.hms.harvard.edu/projects/n2c2-2010/) and "
                f"place the three release tarballs under {tar.parent} before "
                "running this script."
            ) from exc

    LOGGER.info("Extracting %s", args.train_tar.name)
    train_root = safe_extract_tar(args.train_tar, args.work_dir / "train")
    LOGGER.info("Extracting %s", args.test_data_tar.name)
    test_txt_root = safe_extract_tar(args.test_data_tar, args.work_dir / "test_txt")
    LOGGER.info("Extracting %s", args.test_ref_tar.name)
    test_ast_root = safe_extract_tar(args.test_ref_tar, args.work_dir / "test_ast")

    train_csv = args.out_dir / "2010_train.csv"
    test_csv = args.out_dir / "2010_test.csv"

    n_train = process_split(train_pairs(train_root), train_csv)
    LOGGER.info("Wrote %s rows to %s", n_train, train_csv)

    n_test = process_split(test_pairs(test_txt_root, test_ast_root), test_csv)
    LOGGER.info("Wrote %s rows to %s", n_test, test_csv)

    if args.cleanup:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        LOGGER.info("Removed %s", args.work_dir)


if __name__ == "__main__":
    main()
