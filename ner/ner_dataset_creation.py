"""Recreate the historical service-balanced NER note sample.

This one-off script was used during paper preparation to choose the MIMIC-IV
discharge summaries that were sent for manual NER annotation. It samples
additional MIMIC-IV-Note discharge summaries by ``curr_service`` in proportion
to the overall MIMIC-IV discharge-summary service distribution, so the note
pool covers medicine, surgery, psychiatry, and other service types according
to their source-data mix.

The SNOMED CT Entity Linking Challenge note set is used as a coverage boundary:
candidate notes are excluded if they already appear in that challenge, which
makes the new entity dataset cover different MIMIC notes. The script does not
try to quality-check against SNOMED annotations because that challenge uses a
different schema.

Inputs:
  - MIMIC-IV ``hosp/services.csv`` with ``subject_id``, ``hadm_id``, and
    ``curr_service`` columns.
  - MIMIC-IV-Note ``discharge.csv`` with ``note_id``, ``subject_id``,
    ``hadm_id``, ``note_type``, and ``text`` columns.
  - SNOMED CT Entity Linking Challenge ``mimic-iv_notes_training_set.csv`` with
    a ``note_id`` column.

Outputs default to local-only files under ``data/ner/``:
  - ``ner_dataset_notes.csv``
  - ``ner_dataset_notes.parquet``

The paper-era defaults use MIMIC-IV v2.2, MIMIC-IV-Note v2.2, and version
1.0.0 of the SNOMED CT Entity Linking Challenge.

Example:
    python ner/ner_dataset_creation.py
    python ner/ner_dataset_creation.py \\
        --services data/mimic-iv-note/services.csv \\
        --notes data/mimic-iv-note/discharge.csv \\
        --snomed-notes data/mimic-iv-note/mimic-iv_notes_training_set.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ner_dataset_creation")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MIMIC_NOTE_DIR = REPO_ROOT / "data" / "mimic-iv-note"
DEFAULT_SERVICES = DEFAULT_MIMIC_NOTE_DIR / "services.csv"
DEFAULT_NOTES = DEFAULT_MIMIC_NOTE_DIR / "discharge.csv"
DEFAULT_SNOMED_NOTES = DEFAULT_MIMIC_NOTE_DIR / "mimic-iv_notes_training_set.csv"
DEFAULT_OUTPUT_STEM = REPO_ROOT / "data" / "ner" / "ner_dataset_notes"

DEFAULT_TARGET_TOTAL_SAMPLES = 389
DEFAULT_RANDOM_STATE = 1
SURGICAL_SERVICES = [
    "CSURG",
    "NSURG",
    "ORTHO",
    "PSURG",
    "SURG",
    "TRAUM",
    "TSURG",
    "VSURG",
    "ENT",
]

REQUIRED_SERVICE_COLUMNS = {"subject_id", "hadm_id", "curr_service"}
REQUIRED_NOTE_COLUMNS = {"note_id", "subject_id", "hadm_id", "note_type", "text"}
REQUIRED_SNOMED_COLUMNS = {"note_id"}
OUTPUT_COLUMNS = ["note_id", "subject_id", "hadm_id", "curr_service", "text"]


def _configure_stdio() -> None:
    """Use UTF-8 streams on Windows terminals when Python exposes reconfigure()."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _validate_columns(frame: pd.DataFrame, required: set[str], source: str | Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required column(s): {', '.join(sorted(missing))}")


def _read_csv(path: Path, required_columns: set[str], description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. Place the source file under "
            "data/mimic-iv-note/ or pass the corresponding CLI path."
        )
    frame = pd.read_csv(path)
    _validate_columns(frame, required_columns, path)
    LOGGER.info("Read %d rows from %s", len(frame), path)
    return frame


def build_service_balanced_sample(
    services_df: pd.DataFrame,
    all_notes_df: pd.DataFrame,
    snomed_notes_df: pd.DataFrame,
    target_total_samples: int = DEFAULT_TARGET_TOTAL_SAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sampled notes plus service-comparison tables.

    The arithmetic below is the historical manuscript sampling procedure. It
    first estimates the ``curr_service`` distribution in all MIMIC-IV discharge
    summaries, then samples non-SNOMED discharge summaries from each service in
    that same proportion. ``target_total_samples`` is 389 because the ceiling
    step inflates the final sampled set to 400 notes.
    """
    _validate_columns(services_df, REQUIRED_SERVICE_COLUMNS, "services CSV")
    _validate_columns(all_notes_df, REQUIRED_NOTE_COLUMNS, "notes CSV")
    _validate_columns(snomed_notes_df, REQUIRED_SNOMED_COLUMNS, "SNOMED notes CSV")

    snomed_note_ids = snomed_notes_df["note_id"].tolist()

    snomed_included_notes = all_notes_df[all_notes_df["note_id"].isin(snomed_note_ids)]
    snomed_notes_with_service = pd.merge(
        snomed_included_notes,
        services_df,
        on=("subject_id", "hadm_id"),
        how="left",
    ).drop_duplicates("note_id")

    included_counts = snomed_notes_with_service.groupby("curr_service")["note_id"].count()
    included_proportions = included_counts / included_counts.sum()
    included_stats = pd.DataFrame(
        {
            "count_included": included_counts,
            "proportion_included": included_proportions,
        }
    )

    discharge_notes = all_notes_df[all_notes_df["note_type"] == "DS"]
    discharge_notes_with_service = pd.merge(
        discharge_notes,
        services_df,
        on=("subject_id", "hadm_id"),
        how="left",
    ).drop_duplicates("note_id")

    general_counts = discharge_notes_with_service.groupby("curr_service")["note_id"].count()
    general_proportions = general_counts / general_counts.sum()
    general_stats = pd.DataFrame(
        {
            "count_general": general_counts,
            "proportion_general": general_proportions,
        }
    )

    comparison_stats = pd.merge(
        included_stats,
        general_stats,
        left_index=True,
        right_index=True,
        how="outer",
    ).fillna(0)

    LOGGER.info("Comparison of service type counts and proportions:\n%s", comparison_stats)

    prop_surgical_included = comparison_stats.loc[SURGICAL_SERVICES, "proportion_included"].sum()
    prop_surgical_general = comparison_stats.loc[SURGICAL_SERVICES, "proportion_general"].sum()
    LOGGER.info("Proportion of surgical services (SNOMED included): %s", prop_surgical_included)
    LOGGER.info("Proportion of surgical services (general discharge notes): %s", prop_surgical_general)

    comparison_stats["target_count"] = comparison_stats["proportion_general"] * target_total_samples
    comparison_stats["additional_samples_needed"] = comparison_stats["target_count"]
    comparison_stats["additional_samples_needed"] = comparison_stats["additional_samples_needed"].clip(lower=0)
    comparison_stats["additional_samples_needed"] = np.ceil(
        comparison_stats["additional_samples_needed"]
    ).astype(int)

    LOGGER.info(
        "Calculated additional samples needed per service type:\n%s",
        comparison_stats[
            [
                "count_included",
                "proportion_included",
                "count_general",
                "proportion_general",
                "target_count",
                "additional_samples_needed",
            ]
        ],
    )
    LOGGER.info(
        "Total additional samples needed: %d",
        comparison_stats["additional_samples_needed"].sum(),
    )

    sampled_extra_notes_list = []
    for service, samples_needed in comparison_stats["additional_samples_needed"].items():
        if samples_needed > 0:
            service_notes = discharge_notes_with_service[
                (discharge_notes_with_service["curr_service"] == service)
                & (~discharge_notes_with_service["note_id"].isin(snomed_note_ids))
            ]
            sampled_rows = service_notes.sample(
                n=int(samples_needed),
                replace=False if samples_needed <= len(service_notes) else True,
                random_state=random_state,
            )
            sampled_extra_notes_list.append(sampled_rows)

    sampled_extra_notes = pd.concat(sampled_extra_notes_list, ignore_index=True)
    sampled_extra_notes = sampled_extra_notes[OUTPUT_COLUMNS]

    combined_notes_df = pd.concat([snomed_notes_with_service, sampled_extra_notes], ignore_index=True)
    final_counts = combined_notes_df.groupby("curr_service")["note_id"].count()
    final_proportions = final_counts / final_counts.sum()
    proportion_comparison = pd.DataFrame(
        {
            "snomed_proportion": included_proportions,
            "general_proportion": general_proportions,
            "new_dataset_proportion": final_proportions,
        }
    ).fillna(0)

    LOGGER.info("Final comparison of service type proportions:\n%s", proportion_comparison)
    return sampled_extra_notes, comparison_stats, proportion_comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--services",
        type=Path,
        default=DEFAULT_SERVICES,
        help=f"MIMIC-IV hosp/services.csv path (default: {_relative_to_repo(DEFAULT_SERVICES)})",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_NOTES,
        help=f"MIMIC-IV-Note discharge.csv path (default: {_relative_to_repo(DEFAULT_NOTES)})",
    )
    parser.add_argument(
        "--snomed-notes",
        type=Path,
        default=DEFAULT_SNOMED_NOTES,
        help=(
            "SNOMED CT Entity Linking Challenge note-list CSV "
            f"(default: {_relative_to_repo(DEFAULT_SNOMED_NOTES)})"
        ),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output path without suffix (default: {_relative_to_repo(DEFAULT_OUTPUT_STEM)})",
    )
    parser.add_argument(
        "--target-total-samples",
        type=int,
        default=DEFAULT_TARGET_TOTAL_SAMPLES,
        help=(
            "Historical target before ceiling by service type "
            f"(default: {DEFAULT_TARGET_TOTAL_SAMPLES}; yields 400 notes with the paper data)"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"pandas sampling seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    services_path = _resolve_repo_path(args.services)
    notes_path = _resolve_repo_path(args.notes)
    snomed_notes_path = _resolve_repo_path(args.snomed_notes)
    output_stem = _resolve_repo_path(args.output_stem)

    try:
        LOGGER.info("Reading services: %s", services_path)
        services_df = _read_csv(services_path, REQUIRED_SERVICE_COLUMNS, "services CSV")
        LOGGER.info("Reading notes: %s", notes_path)
        all_notes_df = _read_csv(notes_path, REQUIRED_NOTE_COLUMNS, "MIMIC-IV-Note discharge CSV")
        LOGGER.info("Reading SNOMED note list: %s", snomed_notes_path)
        snomed_notes_df = _read_csv(
            snomed_notes_path,
            REQUIRED_SNOMED_COLUMNS,
            "SNOMED CT Entity Linking Challenge note-list CSV",
        )
        sampled_extra_notes, _, _ = build_service_balanced_sample(
            services_df,
            all_notes_df,
            snomed_notes_df,
            target_total_samples=args.target_total_samples,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    parquet_path = output_stem.with_suffix(".parquet")
    sampled_extra_notes.to_csv(csv_path, index=False)
    sampled_extra_notes.to_parquet(parquet_path, index=False)
    LOGGER.info("Wrote %d sampled notes to %s", len(sampled_extra_notes), csv_path)
    LOGGER.info("Wrote %d sampled notes to %s", len(sampled_extra_notes), parquet_path)


if __name__ == "__main__":
    main()
