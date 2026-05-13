#!/usr/bin/env python3
"""
End-to-end orchestrator for the Entity-Coding pipeline.

Runs the three stages of the paper in sequence on a file of clinical notes:

1. Entity extraction with assertion filtering (``ner/extract_entities.py``):
   detects clinical entities and drops those marked as absent (negated),
   hypothetical, etc., producing an entities CSV.
2. ICD-10 prediction with evidence
   (``external/plm_ca/infer_with_explanations.py``): runs the entity-only
   PLM-CA model and computes per-entity attributions for each predicted code.
3. Optional per-note HTML visualisation
   (``code_evidence/visualise_predictions_explanations.py``).

Inputs: a CSV/parquet with ``note_id, text`` columns, or a single free-text
note passed inline with ``--text "..."`` for ad-hoc exploration. The inline
form is written internally to ``results/ner/freeform.csv`` (note_id
``freeform``) and then driven through the same pipeline; entity extraction is
forced so repeated ``--text`` runs do not skip via the resume cache.

Outputs: an entities CSV and ICD-code predictions (``.csv`` + ``.parquet``).
Two independent visualisation flags add HTML side-outputs:

- ``--visualize-entities`` writes per-note entity/assertion HTMLs under
  ``results/ner/docs_with_ner/`` (displaCy spans).
- ``--visualize-evidence`` writes per-note ICD prediction + evidence HTMLs
  under ``results/visualised_notes/``.

Examples:
    python run_pipeline.py data/sample_data/sample_notes.csv \\
        --visualize-entities --visualize-evidence
    python run_pipeline.py --text "Patient presents with chest pain and \\
        shortness of breath." --visualize-evidence
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Force UTF-8 on stdio so unicode renders on Windows (cp1252) too.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOGGER = logging.getLogger("run_pipeline")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
FORMATTED_TEXTS_DIR = REPO_ROOT / "results" / "formatted_texts"
VISUALISED_NOTES_DIR = REPO_ROOT / "results" / "visualised_notes"
REQUIRED_INPUT_COLUMNS = ("note_id", "text")

# Ad-hoc free-text input: single-row CSV materialised on each --text invocation.
FREEFORM_NOTE_ID = "freeform"
FREEFORM_INPUT_PATH = REPO_ROOT / "results" / "ner" / f"{FREEFORM_NOTE_ID}.csv"


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env == "entitycoding":
        LOGGER.info("Using conda environment: %s", env)
        return
    LOGGER.warning(
        "Not in 'entitycoding' conda environment (current: %s). "
        "Continuing -- proceed at your own risk if required packages are missing.",
        env or "none",
    )


def peek_input_columns(path: Path) -> set:
    """Return the column names of a CSV/parquet without loading row data."""
    if path.suffix.lower() == ".parquet":
        return set(pq.read_schema(str(path)).names)
    return set(pd.read_csv(path, nrows=0).columns)


def materialise_text_input(text: str) -> Path:
    """Write a single ad-hoc note to a one-row CSV and return the resolved path.

    Used when the user passes ``--text "..."`` instead of a CSV. Always writes
    to the same fixed location so downstream output filenames are predictable;
    the caller forces re-extraction so this stable note_id does not get
    short-circuited by entity extraction's resume cache on repeat runs.
    """
    FREEFORM_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"note_id": FREEFORM_NOTE_ID, "text": text}]).to_csv(
        FREEFORM_INPUT_PATH, index=False
    )
    LOGGER.info("Wrote ad-hoc text (%d chars) to %s", len(text), FREEFORM_INPUT_PATH)
    return FREEFORM_INPUT_PATH.resolve()


def validate_input_file(path):
    p = Path(path)
    if not p.exists():
        LOGGER.error("Input file not found: %s", p)
        return False
    if p.suffix.lower() not in {".parquet", ".csv"}:
        LOGGER.error("Unsupported file format %s (expected .parquet or .csv)", p.suffix)
        return False
    try:
        columns = peek_input_columns(p)
    except Exception as e:
        LOGGER.error("Could not read column header from %s: %s", p, e)
        return False
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
    if missing:
        LOGGER.error(
            "Input %s is missing required columns: %s. Found columns: %s",
            p, missing, sorted(columns),
        )
        return False
    LOGGER.info("Input file: %s (%.1f MB)", p, p.stat().st_size / (1024 * 1024))
    return True


def run_step(name, cmd, cwd=None):
    """Run a subprocess step, streaming output. Returns True on success."""
    LOGGER.info("Step: %s", name)
    LOGGER.info("$ %s%s", " ".join(cmd[1:]), f"   (cwd={cwd})" if cwd else "")
    start = time.time()
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error("%s failed (exit code %d)", name, e.returncode)
        return False
    LOGGER.info("%s completed in %.1fs", name, time.time() - start)
    return True


def run_entity_extraction(
    input_file, entities_file, max_workers,
    save_formatted, save_ner_docs, force, log_level,
):
    cmd = [
        sys.executable, str(REPO_ROOT / "ner" / "extract_entities.py"),
        str(input_file),
        "--output_file", str(entities_file),
        "--max_workers", str(max_workers),
        "--log-level", log_level,
    ]
    if save_ner_docs:
        cmd.append("--save-ner-docs")
    if save_formatted:
        cmd.append("--save-formatted-texts")
    if force:
        cmd.append("--force")
    if not run_step("Entity Extraction", cmd, cwd=str(REPO_ROOT)):
        return False
    if not Path(entities_file).exists():
        LOGGER.error("Entity extraction did not produce %s", entities_file)
        return False
    return True


def run_icd_coding(entities_file, output_prefix):
    """Run PLM-CA inference. cwd switches to ``external/plm_ca`` (model paths
    resolve relative to that directory); inputs/outputs are passed absolute."""
    plm_ca_dir = REPO_ROOT / "external" / "plm_ca"
    cmd = [
        sys.executable, "infer_with_explanations.py",
        str(Path(entities_file).resolve()),
        str(Path(output_prefix).resolve()),
    ]
    if not run_step("ICD Coding", cmd, cwd=str(plm_ca_dir)):
        return False
    csv_out = Path(f"{output_prefix}.csv")
    parquet_out = Path(f"{output_prefix}.parquet")
    if not csv_out.exists() or not parquet_out.exists():
        LOGGER.error("ICD coding did not produce %s and %s", csv_out, parquet_out)
        return False
    return True


def run_visualization(icd_results_file):
    if not FORMATTED_TEXTS_DIR.exists():
        LOGGER.error(
            "Visualization requires %s/. Re-run with --visualize-evidence "
            "(or extract_entities --save-formatted-texts).",
            FORMATTED_TEXTS_DIR,
        )
        return False
    if not Path(icd_results_file).exists():
        LOGGER.error("ICD results file not found: %s", icd_results_file)
        return False
    cmd = [
        sys.executable, str(REPO_ROOT / "code_evidence" / "visualise_predictions_explanations.py"),
        str(icd_results_file),
    ]
    if not run_step("Visualization", cmd, cwd=str(REPO_ROOT)):
        return False
    html_files = sorted(VISUALISED_NOTES_DIR.glob("*.html")) if VISUALISED_NOTES_DIR.exists() else []
    if not html_files:
        LOGGER.error("No visualization files were generated")
        return False
    LOGGER.info("Generated %d HTML visualization files in %s/", len(html_files), VISUALISED_NOTES_DIR)
    return True


def log_summary(input_file, entities_file, output_prefix, total_time):
    LOGGER.info("Pipeline completed in %.1fs", total_time)
    LOGGER.info("  Input:    %s", input_file)
    LOGGER.info("  Entities: %s", entities_file)
    LOGGER.info("  Codes:    %s.{csv,parquet}", output_prefix)
    try:
        entities_df = pd.read_csv(entities_file)
        results_df = pd.read_csv(f"{output_prefix}.csv")
        LOGGER.info(
            "  Notes: %d  entities: %d  ICD codes predicted: %d",
            entities_df["note_id"].nunique(),
            len(entities_df),
            len(results_df),
        )
    except Exception as e:
        LOGGER.warning("Could not load result statistics: %s", e)


def resolve_outputs(input_file, output_prefix):
    """Compute (entities_file, output_prefix) from CLI args.

    Default layout (no prefix supplied) places artefacts under ``results/ner/``
    and ``results/coded/`` keyed off the input basename. A user-supplied prefix
    has any trailing ``.csv`` / ``.parquet`` stripped before being suffixed.
    """
    if output_prefix is None:
        stem = Path(input_file).stem
        return (
            str(REPO_ROOT / "results" / "ner" / f"{stem}_entities.csv"),
            str(REPO_ROOT / "results" / "coded" / f"{stem}_results"),
        )
    for ext in (".csv", ".parquet"):
        if output_prefix.endswith(ext):
            output_prefix = output_prefix[: -len(ext)]
            break
    return f"{output_prefix}_entities.csv", f"{output_prefix}_icd_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Entity-Coding pipeline: NER + AC -> ICD coding -> visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py data/sample_data/sample_notes.csv\n"
            "  python run_pipeline.py data/notes.csv my_results --max_workers 2\n"
            "  python run_pipeline.py data/sample_data/sample_notes.csv \\\n"
            "      --visualize-entities --visualize-evidence\n"
            "  python run_pipeline.py --text \"Patient presents with chest pain.\" \\\n"
            "      --visualize-evidence\n"
        ),
    )
    parser.add_argument("input_file", nargs="?", default=None,
        help="Path to input file (.parquet or .csv, must contain 'note_id' and 'text' columns). "
             "Omit when supplying --text.")
    parser.add_argument("output_prefix", nargs="?", default=None,
        help="Output file prefix (default: derived from input filename)")
    parser.add_argument("--text", default=None,
        help="Run the pipeline on a single inline note instead of a file. The string is "
             "written to results/ner/freeform.csv with note_id 'freeform' and then driven "
             "through the normal flow; entity extraction is forced for this mode.")
    parser.add_argument("--max_workers", type=int, default=2,
        help="Parallel workers for entity extraction (default: 2)")
    parser.add_argument("--visualize-entities", action="store_true",
        help="Save per-note HTML files highlighting detected entities and assertion "
             "statuses under results/ner/docs_with_ner/ (via displaCy).")
    parser.add_argument("--visualize-evidence", action="store_true",
        help="Save per-note HTML files showing predicted ICD codes and the entity "
             "evidence supporting each code under results/visualised_notes/. "
             "Implies --save-formatted-texts upstream so the visualiser has the "
             "formatted note bodies it needs.")
    parser.add_argument("--force", action="store_true",
        help="Reprocess every document from scratch, ignoring existing entity-extraction output. "
             "Without this flag, notes already present in the entities CSV are skipped.")
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS,
        help="Logging verbosity for this script and entity extraction (default: INFO).")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level)

    if (args.input_file is None) == (args.text is None):
        LOGGER.error(
            "Supply exactly one of: positional input_file or --text \"...\". "
            "Got input_file=%r, text=%s.",
            args.input_file, "<set>" if args.text is not None else "<unset>",
        )
        sys.exit(2)

    check_conda_env()
    if args.text is not None:
        input_file = materialise_text_input(args.text)
        force_extraction = True
    else:
        input_file = Path(args.input_file).resolve()
        if not validate_input_file(input_file):
            sys.exit(1)
        force_extraction = args.force

    entities_file, output_prefix = resolve_outputs(input_file, args.output_prefix)
    entities_file = str(Path(entities_file).resolve())
    output_prefix = str(Path(output_prefix).resolve())
    LOGGER.info(
        "Config: max_workers=%d  visualize_entities=%s  visualize_evidence=%s  force=%s",
        args.max_workers, args.visualize_entities, args.visualize_evidence, force_extraction,
    )
    LOGGER.info("  entities -> %s", entities_file)
    LOGGER.info("  codes    -> %s.{csv,parquet}", output_prefix)

    for path in (entities_file, f"{output_prefix}.csv"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # --visualize-evidence implies --save-formatted-texts upstream because the
    # evidence visualiser later reads results/formatted_texts/.
    if not run_entity_extraction(
        input_file, entities_file, args.max_workers,
        save_formatted=args.visualize_evidence,
        save_ner_docs=args.visualize_entities,
        force=force_extraction,
        log_level=args.log_level,
    ):
        sys.exit(1)

    if not run_icd_coding(entities_file, output_prefix):
        sys.exit(1)

    if args.visualize_evidence and not run_visualization(f"{output_prefix}.csv"):
        LOGGER.warning("Pipeline completed but visualisation failed; main results still available")

    log_summary(input_file, entities_file, output_prefix, time.time() - start)


if __name__ == "__main__":
    main()
