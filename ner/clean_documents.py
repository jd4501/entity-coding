"""Clean note text with the same preprocessing used before entity extraction.

This is the no-model entry point for the NER side of the project. It reads a
CSV or parquet file with ``note_id`` and ``text`` columns, runs the document
cleaning rules used by ``ner/extract_entities.py``, and writes a CSV with
``note_id,text``. The output quotes all fields so embedded newlines in cleaned
notes round-trip through ``pandas.read_csv``.

Use ``--filter_ids`` to keep only a listed set of notes, ``--max_workers`` to
control CPU parallelism, and ``--force`` to discard an existing partial output
instead of resuming from it.

Example:
    python ner/clean_documents.py data/sample_data/sample_notes.csv \\
        --output_file results/ner/sample_cleaned_notes.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("clean_documents")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_NOTE_COLUMNS = {"note_id", "text"}

# Ensure the ``ner`` directory is importable when invoked as
# ``python ner/clean_documents.py``. Spawned workers inherit this path setup.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# Worker-local pipeline built once per process and reused across documents.
_worker_nlp = None
_build_cleaning_nlp = None
_clean_document_text = None


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


def _validate_columns(frame: pd.DataFrame, required: set[str], source: str | Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source} missing required column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {list(frame.columns)}"
        )


def _read_notes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix} (expected .csv or .parquet)")

    _validate_columns(frame, REQUIRED_NOTE_COLUMNS, path)
    return frame.loc[:, ["note_id", "text"]]


def _read_filter_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Filter ID file not found: {path}")

    frame = pd.read_csv(path)
    if "note_id" in frame.columns:
        ids = frame["note_id"]
    elif len(frame.columns) > 0:
        ids = frame.iloc[:, 0]
    else:
        raise ValueError(f"{path} has no columns; expected a note_id column or first-column IDs")
    return set(ids.astype(str).tolist())


def _load_cleaning_helpers():
    global _build_cleaning_nlp, _clean_document_text
    if _build_cleaning_nlp is None or _clean_document_text is None:
        from document_cleaning import build_cleaning_nlp, clean_document_text

        _build_cleaning_nlp = build_cleaning_nlp
        _clean_document_text = clean_document_text
    return _build_cleaning_nlp, _clean_document_text


def _init_worker() -> None:
    global _worker_nlp
    build_cleaning_nlp, _ = _load_cleaning_helpers()
    _worker_nlp = build_cleaning_nlp()


def _clean_one(note_id: object, text: str) -> tuple[object, str]:
    global _worker_nlp
    build_cleaning_nlp, clean_document_text = _load_cleaning_helpers()
    if _worker_nlp is None:
        _worker_nlp = build_cleaning_nlp()
    cleaned = clean_document_text(text, nlp=_worker_nlp)
    return note_id, cleaned


def clean_documents(
    input_path: Path,
    output_path: Path,
    filter_ids_path: Path | None = None,
    max_workers: int = 4,
    force: bool = False,
) -> None:
    """Clean documents in parallel and append results to ``output_path``.

    Existing outputs are treated as resumable partial runs. This mirrors the
    project entity-extraction script so long MIMIC cleaning jobs can continue
    after interruption without rewriting notes that are already present.
    """
    if max_workers < 1:
        raise ValueError(f"--max_workers must be at least 1 (got {max_workers})")

    docs = _read_notes(input_path)
    LOGGER.info("Read %d notes from %s", len(docs), input_path)

    if filter_ids_path:
        wanted = _read_filter_ids(filter_ids_path)
        LOGGER.info("Filter list has %d note_ids", len(wanted))
        before = len(docs)
        docs = docs[docs["note_id"].astype(str).isin(wanted)]
        missing = wanted - set(docs["note_id"].astype(str).tolist())
        LOGGER.info(
            "Kept %d/%d notes after filter; %d requested note_ids were not found in input.",
            len(docs),
            before,
            len(missing),
        )
        if missing:
            LOGGER.warning("Example missing note_ids: %s", sorted(missing)[:5])

    if docs.empty:
        LOGGER.info("No notes to process; exiting.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_note_ids: set[str] = set()
    if force and output_path.exists():
        output_path.unlink()
        LOGGER.info("--force: removed existing output, reprocessing all documents.")
    elif output_path.exists() and output_path.stat().st_size > 0:
        try:
            previous = pd.read_csv(output_path, usecols=["note_id"])
        except ValueError as exc:
            raise ValueError(
                f"Existing output {output_path} does not contain a note_id column; "
                "pass --force to replace it."
            ) from exc
        processed_note_ids.update(previous["note_id"].astype(str).tolist())
        LOGGER.info(
            "Resuming: %d notes already present in %s (pass --force to reprocess).",
            len(processed_note_ids),
            output_path,
        )

    mp_context = multiprocessing.get_context("spawn")
    total = len(docs)
    processed = 0
    skipped = 0
    failed = 0
    start = time.time()

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writerow(["note_id", "text"])

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_worker,
        ) as executor:
            futures = []
            for row in docs.itertuples(index=False):
                note_id = str(row.note_id)
                if note_id in processed_note_ids:
                    skipped += 1
                    continue
                futures.append(executor.submit(_clean_one, row.note_id, row.text))

            if not futures:
                LOGGER.info("No unprocessed notes remain; output is up to date: %s", output_path)

            for future in concurrent.futures.as_completed(futures):
                try:
                    note_id, cleaned = future.result()
                except Exception as exc:
                    failed += 1
                    LOGGER.error(
                        "Error cleaning a document: %s",
                        exc,
                        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                    )
                    continue

                writer.writerow([note_id, cleaned])
                f.flush()
                processed += 1
                if processed % 100 == 0 or processed == len(futures):
                    LOGGER.info(
                        "Cleaned %d/%d (skipped %d already processed) [%.1fs elapsed]",
                        processed,
                        len(futures),
                        skipped,
                        time.time() - start,
                    )

    if failed:
        LOGGER.warning("%d document(s) failed during cleaning and were not written.", failed)
    LOGGER.info(
        "Done. Wrote %d cleaned notes to %s (%d skipped, %d in scope).",
        processed,
        output_path,
        skipped,
        total,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input notes file (.csv or .parquet) with note_id and text columns.",
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        dest="output_file",
        type=Path,
        required=True,
        help="Output CSV path. Will contain columns note_id and text.",
    )
    parser.add_argument(
        "--filter_ids",
        "--filter-ids",
        dest="filter_ids",
        type=Path,
        default=None,
        help=(
            "Optional CSV listing note_ids to keep. If the file has a note_id "
            "column it is used; otherwise the first column is read."
        ),
    )
    parser.add_argument(
        "--max_workers",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing output file and reprocess from scratch.",
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

    try:
        clean_documents(
            input_path=args.input_file,
            output_path=args.output_file,
            filter_ids_path=args.filter_ids,
            max_workers=args.max_workers,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
