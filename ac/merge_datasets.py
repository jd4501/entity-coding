"""Merge the four AC sources into stratified train, val, and test splits.

By default this script consumes the CSVs regenerated under `ac/data/` by the
four `prepare_*` scripts:

    2010_train.csv and 2010_test.csv
    i2b2_2012_merged.csv
    mimic_assertion_data.csv
    our_new_assertions.csv

It writes `train_expanded.csv`, `val_expanded.csv`, and `test_expanded.csv`
back to the same directory. The label handling matches the paper pipeline:
i2b2 2010 `conditional` rows are dropped, and `present` rows from MIMIC-III
and i2b2 2012 are dropped because the 2010 source already supplies many
present examples.

Pass `--deterministic` to sort by normalized text before each stratified split.
That option is useful when source-row order is not stable across reruns, but
the default path (no flag) reproduces the splits from the original paper.
"""

from __future__ import annotations

import argparse
import logging
import re
import string
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from common import (
        AC_DATA_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_file,
    )
except ModuleNotFoundError:
    from ac.common import (
        AC_DATA_DIR,
        add_log_level,
        configure_logging,
        require_columns,
        require_file,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOGGER = logging.getLogger("merge_datasets")
REQUIRED_COLUMNS = {"text", "assertion"}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sort_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by normalized text and assertion before splitting."""
    if "text_norm" not in df.columns:
        df = df.assign(text_norm=df["text"].map(normalize_text))
    return df.sort_values(["text_norm", "assertion"], kind="stable").reset_index(drop=True)


def read_assertion_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = require_file(data_dir / filename, f"AC source CSV {filename}")
    df = pd.read_csv(path)
    require_columns(df.columns, REQUIRED_COLUMNS, path)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=AC_DATA_DIR,
        help="Directory containing source CSVs and receiving output splits.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Sort by normalized text before each stratified split. This makes outputs "
            "depend on row content instead of source ordering, but does not match "
            "the frozen paper splits exactly."
        ),
    )
    add_log_level(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    sort_for_split = sort_canonical if args.deterministic else (lambda df: df.reset_index(drop=True))

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    base_train = read_assertion_csv(data_dir, "2010_train.csv")
    base_train = base_train[base_train["assertion"] != "conditional"]
    base_test = read_assertion_csv(data_dir, "2010_test.csv")
    base_test = base_test[base_test["assertion"] != "conditional"]

    mimic = read_assertion_csv(data_dir, "mimic_assertion_data.csv")
    mimic = mimic[mimic["assertion"] != "present"]

    i2b2_2012 = read_assertion_csv(data_dir, "i2b2_2012_merged.csv")
    i2b2_2012 = i2b2_2012[i2b2_2012["assertion"] != "present"].copy()
    i2b2_2012["text"] = i2b2_2012["text"].apply(unescape)

    for df in (base_train, base_test, i2b2_2012):
        df["text_norm"] = df["text"].apply(normalize_text)

    i2b2_2012 = i2b2_2012[~i2b2_2012["text_norm"].isin(base_train["text_norm"])]
    i2b2_2012 = i2b2_2012[~i2b2_2012["text_norm"].isin(base_test["text_norm"])]

    base_train = base_train.assign(source="train")
    base_test = base_test.assign(source="test")
    combined_df = pd.concat([base_train, base_test], ignore_index=True).drop_duplicates(subset="text")

    target_col = "assertion"
    total_size = len(combined_df)
    test_size_abs = int(total_size * 0.2)
    train_size_abs = int(total_size * 0.7)
    val_size_abs = total_size - train_size_abs - test_size_abs

    test_src = combined_df[combined_df["source"] == "test"]
    train_src = combined_df[combined_df["source"] == "train"]

    sorted_test_src = sort_for_split(test_src)
    test_data_from_test, remaining_test = train_test_split(
        sorted_test_src,
        test_size=(len(base_test) - test_size_abs),
        stratify=sorted_test_src[target_col],
        random_state=args.seed,
    )

    remaining_data = pd.concat(
        [train_src[["text", target_col]], remaining_test[["text", target_col]]],
        ignore_index=True,
    )
    remaining_data = sort_for_split(remaining_data)

    train_data, val_data = train_test_split(
        remaining_data,
        test_size=val_size_abs / (train_size_abs + val_size_abs),
        stratify=remaining_data[target_col],
        random_state=args.seed,
    )

    LOGGER.info("Total size for i2b2 2010 rebalanced pool: %s", total_size)
    LOGGER.info("Train size for i2b2 2010 rebalanced pool: %s", len(train_data))
    LOGGER.info("Validation size for i2b2 2010 rebalanced pool: %s", len(val_data))
    LOGGER.info("Test size for i2b2 2010 rebalanced pool: %s", len(test_data_from_test))

    train_data = train_data[["text", "assertion"]]
    val_data = val_data[["text", "assertion"]]
    test_data_from_test = test_data_from_test[["text", "assertion"]]

    i2b2_2012 = i2b2_2012[["text", "assertion"]]
    expanded = pd.concat([mimic, i2b2_2012], ignore_index=True)

    combined_df["text_norm"] = combined_df["text"].apply(normalize_text)
    expanded["text"] = expanded["text"].apply(unescape)
    expanded["text_norm"] = expanded["text"].apply(normalize_text)
    expanded = expanded.drop_duplicates(subset="text_norm")
    expanded = expanded[~expanded["text_norm"].isin(set(combined_df["text_norm"]))]

    new_data = read_assertion_csv(data_dir, "our_new_assertions.csv")
    expanded = pd.concat([expanded[["text", "assertion"]], new_data], ignore_index=True).dropna()
    expanded = sort_for_split(expanded)

    expanded_train, expanded_testval = train_test_split(
        expanded,
        test_size=0.3,
        stratify=expanded[target_col],
        random_state=args.seed,
    )
    expanded_testval = sort_for_split(expanded_testval)
    expanded_test, expanded_val = train_test_split(
        expanded_testval,
        test_size=0.65,
        stratify=expanded_testval[target_col],
        random_state=args.seed,
    )

    final_train = pd.concat([train_data, expanded_train[["text", "assertion"]]], ignore_index=True)
    final_val = pd.concat([val_data, expanded_val[["text", "assertion"]]], ignore_index=True)
    final_test = pd.concat([test_data_from_test, expanded_test[["text", "assertion"]]], ignore_index=True)

    train_out = data_dir / "train_expanded.csv"
    val_out = data_dir / "val_expanded.csv"
    test_out = data_dir / "test_expanded.csv"
    final_train.to_csv(train_out, index=False)
    final_val.to_csv(val_out, index=False)
    final_test.to_csv(test_out, index=False)

    total = len(final_train) + len(final_val) + len(final_test)
    LOGGER.info("Total expanded size: %s", total)
    LOGGER.info("Train size: %s", len(final_train))
    LOGGER.info("Validation size: %s", len(final_val))
    LOGGER.info("Test size: %s", len(final_test))
    LOGGER.info("Wrote %s, %s, and %s", train_out, val_out, test_out)


if __name__ == "__main__":
    main()
