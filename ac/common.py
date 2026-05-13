"""Shared helpers for the assertion classification scripts.

The AC pipeline reads gated source corpora from local staging directories and
writes regenerated CSVs under `ac/data/`. These helpers keep path anchors,
logging setup, file checks, and tar extraction consistent across the scripts
without changing the paper reproduction flow.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from collections.abc import Iterable
from pathlib import Path


AC_DIR = Path(__file__).resolve().parent
REPO_ROOT = AC_DIR.parent
AC_DATA_DIR = AC_DIR / "data"
AC_SOURCES_DIR = AC_DIR / "sources"
REPO_DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_DATA_DIR / "models"
ENTITYCODING_RELEASE_DIR = REPO_DATA_DIR / "mimic-iv-ext-entitycoding"
BASE_ROBERTA_PM_DIR = (
    MODELS_DIR
    / "RoBERTa-base-PM-M3-Voc-distill-align-hf"
    / "RoBERTa-base-PM-M3-Voc-distill-align"
    / "RoBERTa-base-PM-M3-Voc-distill-align-hf"
)

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity.",
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format=LOG_FORMAT)


def progress_disabled(logger: logging.Logger) -> bool:
    return not logger.isEnabledFor(logging.INFO)


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    return path


def require_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {path}")
    return path


def require_columns(columns: Iterable[str], required: set[str], source: Path) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def safe_extract_tar(tar_path: Path, dest: Path) -> Path:
    """Extract a gzip tarball after checking for path traversal and links."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (dest / member.name).resolve()
            if not _is_within(target, dest_root):
                raise ValueError(f"Unsafe tar member path in {tar_path}: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing tar link member in {tar_path}: {member.name}")
        tf.extractall(dest, members=members)

    top_levels = {
        member.name.split("/", 1)[0]
        for member in members
        if member.name and not member.name.startswith("./")
    }
    if len(top_levels) == 1:
        return dest / next(iter(top_levels))
    return dest
