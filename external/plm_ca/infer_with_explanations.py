"""Predict ICD-10 codes from entity-only documents and extract evidence.

This is one half of the pair of inference scripts in this directory. Both
arms run PLM-CA (the AttInGrad-explainable PLM-ICD variant from Edin et al.,
2024); they differ only in what the model was trained on:
  - infer_with_explanations.py (this file): the *entity-only* arm. Runs
    the PLM-CA checkpoint trained on consolidated entity documents and emits
    evidence at the *line* level (one line == one entity span, joined with
    newlines as the model saw them).
  - infer_with_explanations_fulltext.py: the *full-text* arm. Runs the
    PLM-CA checkpoint trained on raw discharge summaries (the project's
    full-text comparator) and emits evidence at the *word* level.
    merge_contiguous_spans.py then collapses adjacent words into contiguous
    character spans for downstream evaluation.

Both arms are wired into code_evidence/code_evidence_eval.ipynb for the MDACE
evidence-overlap evaluation reported in the manuscript (Tables 8 and 12).

Inputs:
  - input_file: CSV with columns note_id, start_index, end_index, text, in
    the schema produced by ner/extract_entities.py. Each row is a detected
    entity, with text already wrapped in its type token (e.g. ``<disorder>
    pneumonia``).

Outputs:
  - output_file.csv and output_file.parquet, one row per (note_id,
    predicted_code) pair, with columns note_id, predicted_code,
    predicted_code_probability, evidence_line_numbers, evidence_spans,
    evidence_texts, evidence_attributions. Evidence lists are ordered by
    descending per-line attribution.

Usage (from external/plm_ca/, with the entitycoding conda env active):

    python infer_with_explanations.py <input_file> [<output_file>]

The input file is read as CSV; pass it with or without the .csv suffix
(.parquet inputs are rejected). The output argument is an output prefix
that may be passed with or without a .csv/.parquet suffix (any known
suffix is stripped, and both .csv and .parquet are written side-by-side).
The decision boundary defaults to the paper-tuned value for the
entity-only model. The per-token attribution cutoff defaults to a
permissive value so general-purpose inference keeps essentially all
line-level evidence; the tighter paper-tuned cutoff used for MDACE
evaluation is applied inside code_evidence/code_evidence_eval.ipynb. Both
can be overridden with --decision-boundary / --attr-threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.progress import track
from transformers import AutoTokenizer

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Special tokens have been added")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from explainable_medical_coding.utils.loaders import load_trained_model  # noqa: E402
from explainable_medical_coding.explainability.explanation_methods import (  # noqa: E402
    get_grad_attention_callable,
)

LOGGER = logging.getLogger("infer_with_explanations")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
ENTITY_MODEL_DIR = SCRIPT_DIR / "models" / "entityonly"
ENTITY_TOKENIZER_DIR = SCRIPT_DIR / "models" / "tokenizer_latest"

# Token budget. Matches data.max_length in models/entityonly/config.yaml.
MAX_LENGTH = 6000

# Paper-tuned defaults. Mirror PROBABILITY_THRESHOLD in
# code_evidence/code_evidence_eval.ipynb when --decision-boundary is left
# unset; see docs/reproduce.md#tuned-thresholds.
DEFAULT_DECISION_BOUNDARY = 0.4040403962135315
DEFAULT_ATTR_THRESHOLD = 0.0001

REQUIRED_INPUT_COLUMNS = ("note_id", "start_index", "end_index", "text")


def _strip_known_suffix(path: str) -> str:
    """Strip a single trailing .csv or .parquet suffix if present."""
    for suffix in (".csv", ".parquet"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _resolve_encoder_path(candidate: str) -> str:
    """Validate the encoder path saved inside the trained model's config.

    All ICD code prediction in this repo (training and inference, entity-only
    and full-text) is initialised from the *base* BioLM RoBERTa-PM variant
    (``RoBERTa-base-PM-M3-Voc-hf``), which is what ``make download_roberta``
    fetches into ``models/roberta-base-pm-m3-voc-hf``. The distilled-aligned
    variant fetched by ``data_download.py --models roberta`` is for NER and
    AC fine-tuning only and is *not* used here.
    """
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not locate the base PM-RoBERTa encoder at '{candidate}'. "
        f"From external/plm_ca/, run `make download_roberta` to fetch "
        f"RoBERTa-base-PM-M3-Voc-hf into models/roberta-base-pm-m3-voc-hf "
        f"(or download "
        f"https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz "
        f"manually and extract it there)."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entity-only ICD inference with AttInGrad evidence extraction. "
            "Companion to infer_with_explanations_fulltext.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd external/plm_ca\n"
            "  python infer_with_explanations.py \\\n"
            "      ../../results/ner/sample_notes_entities.csv \\\n"
            "      ../../results/coded/sample_notes_results"
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Path to the entity CSV from ner/extract_entities.py. "
            "Pass with or without the .csv suffix; .parquet is not accepted "
            "(this script reads CSV only)."
        ),
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=(
            "Output prefix (no suffix; both .csv and .parquet are written). "
            "Default: ../inferred_notes_with_evidence_<input-basename>."
        ),
    )
    parser.add_argument(
        "--decision-boundary",
        type=float,
        default=DEFAULT_DECISION_BOUNDARY,
        help=(
            "Probability threshold above which a code is predicted. Default is "
            "the paper-tuned value for the entity-only model and matches "
            "PROBABILITY_THRESHOLD in code_evidence/code_evidence_eval.ipynb."
        ),
    )
    parser.add_argument(
        "--attr-threshold",
        type=float,
        default=DEFAULT_ATTR_THRESHOLD,
        help=(
            "Per-token attribution cutoff for line-level evidence aggregation. "
            "Tokens with attribution below this are skipped before the per-line "
            "sum. Default catches essentially everything; raise to drop "
            "low-signal lines."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def validate_input_file(filepath: Path) -> pd.DataFrame:
    """Read the entity CSV and confirm it has the columns this script needs."""
    if not filepath.exists():
        LOGGER.error("Input file not found: %s", filepath)
        LOGGER.error(
            "Run entity extraction first, e.g. "
            "`python ner/extract_entities.py <notes.csv> --output_file <entities.csv>`."
        )
        sys.exit(1)

    df = pd.read_csv(filepath)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.error("Input file missing required columns: %s", missing)
        LOGGER.error("Found columns: %s", list(df.columns))
        sys.exit(1)

    LOGGER.info(
        "Input validated: %d entities from %d notes",
        len(df),
        df["note_id"].nunique(),
    )
    return df


def validate_model_files() -> None:
    """Confirm the entity-only checkpoint and tokenizer are present."""
    required = [
        (ENTITY_MODEL_DIR / "target_tokenizer.json", "ICD code mapping"),
        (ENTITY_MODEL_DIR / "config.yaml", "Model configuration"),
        (ENTITY_MODEL_DIR / "best_model.pt", "Trained model weights"),
        (ENTITY_TOKENIZER_DIR, "Entity-aware tokenizer directory"),
    ]
    for path, description in required:
        if not path.exists():
            LOGGER.error("%s not found at %s", description, path)
            LOGGER.error(
                "Run `python data_download.py --models entity-only` from the "
                "repo root to fetch the entity-only checkpoint and tokenizer."
            )
            sys.exit(1)
    LOGGER.debug("All required model files found")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.input_file.endswith(".parquet"):
        LOGGER.error(
            "input_file is .parquet but this script reads CSV only. "
            "Pass the entity CSV emitted by ner/extract_entities.py instead."
        )
        sys.exit(1)

    filename = _strip_known_suffix(args.input_file)
    if args.output_file is not None:
        output_filename = _strip_known_suffix(args.output_file)
    else:
        output_filename = f"../inferred_notes_with_evidence_{os.path.basename(filename)}"

    input_filepath = Path(f"{filename}.csv")
    df = validate_input_file(input_filepath)

    output_csv = Path(f"{output_filename}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    validate_model_files()

    LOGGER.info("Loading entity-only ICD coding model")
    with open(ENTITY_MODEL_DIR / "target_tokenizer.json", "r") as f:
        codes = json.load(f)

    saved_config = OmegaConf.load(ENTITY_MODEL_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # The encoder path baked into the saved config may not exist in this checkout
    # (entity model was trained with a different relative path). Resolve it
    # against the location data_download.py / make download_roberta extracts the
    # PM-RoBERTa tar to.
    saved_config.model.configs.model_path = _resolve_encoder_path(
        saved_config.model.configs.model_path
    )

    text_tokenizer = AutoTokenizer.from_pretrained(str(ENTITY_TOKENIZER_DIR))

    model, _ = load_trained_model(
        ENTITY_MODEL_DIR,
        saved_config,
        pad_token_id=text_tokenizer.pad_token_id,
        device=device,
    )
    model.eval()
    model.to(device)

    # AttInGrad: the attribution method used for the paper. See
    # explainable_medical_coding/explainability/explanation_methods.py for
    # alternative explainers (e.g. plain attention, integrated gradients).
    explainer = get_grad_attention_callable(model)

    grouped = df.groupby("note_id")
    show_progress = LOGGER.isEnabledFor(logging.INFO)
    iterator = (
        track(grouped, description="Processing notes", total=grouped.ngroups)
        if show_progress
        else grouped
    )

    results: list[dict] = []
    decision_boundary = args.decision_boundary
    attr_threshold = args.attr_threshold

    for note_id, group in iterator:
        group_sorted = group.sort_values("start_index").reset_index(drop=True)
        group_sorted["line_number"] = group_sorted.index + 1

        lines = group_sorted["text"].tolist()
        line_spans = []
        pos = 0
        lines_with_newlines = []
        for line in lines:
            line = str(line)
            start_pos = pos
            end_pos = pos + len(line)
            line_spans.append((start_pos, end_pos))
            lines_with_newlines.append(line)
            pos = end_pos + 1  # +1 for the newline character joining lines below

        full_text = "\n".join(lines_with_newlines)

        inputs = text_tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        offset_mapping = inputs["offset_mapping"]

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        predicted_labels = (probs > decision_boundary).nonzero(as_tuple=False)
        target_ids = predicted_labels[:, 1]

        if len(target_ids) == 0:
            continue

        attributions = explainer(input_ids, target_ids, device)
        predicted_probs = probs[0, target_ids].cpu().numpy()
        label_prob_pairs = [
            (tid.item(), predicted_probs[idx], attributions[:, idx])
            for idx, tid in enumerate(target_ids)
        ]
        label_prob_pairs_sorted = sorted(
            label_prob_pairs, key=lambda x: x[1], reverse=True
        )

        if label_prob_pairs_sorted:
            target_ids, predicted_probs, attributions = zip(*label_prob_pairs_sorted)
        else:
            target_ids, predicted_probs, attributions = ([], [], [])

        tokens = text_tokenizer.convert_ids_to_tokens(input_ids[0])
        offsets = offset_mapping[0].tolist()

        for idx, target_id in enumerate(target_ids):
            token_attributions = attributions[idx]
            prob = predicted_probs[idx]
            code = codes[target_id]

            token_info = []
            for token, (start, end), attribution in zip(
                tokens, offsets, token_attributions
            ):
                if start == 0 and end == 0:  # Skip special tokens
                    continue
                token_text = full_text[start:end]
                token_info.append(
                    {
                        "token": token_text,
                        "start": start,
                        "end": end,
                        "attribution": attribution.item(),
                    }
                )

            if not token_info:
                continue

            # Aggregate per-token attributions into per-entity-line totals.
            # Each entity is one newline-joined line in `full_text`, so a
            # subword inside line N contributes its attribution to line N's
            # running sum.
            line_attributions: dict[int, float] = {}
            for t in token_info:
                if t["attribution"] < attr_threshold:
                    continue
                for line_idx, (line_start, line_end) in enumerate(line_spans):
                    if t["start"] >= line_start and t["end"] <= line_end:
                        line_number = line_idx + 1
                        line_attributions[line_number] = (
                            line_attributions.get(line_number, 0.0) + t["attribution"]
                        )
                        break

            if not line_attributions:
                continue

            sorted_lines = sorted(
                line_attributions.items(), key=lambda x: x[1], reverse=True
            )
            evidence_lines = [ln for ln, _ in sorted_lines]
            evidence_attributions = [line_attributions[ln] for ln in evidence_lines]

            evidence_spans = []
            evidence_texts = []
            for line_number in evidence_lines:
                line_row = group_sorted.loc[group_sorted["line_number"] == line_number]
                span_start = line_row["start_index"].values[0]
                span_end = line_row["end_index"].values[0]
                evidence_spans.append((span_start, span_end))
                line_text = line_row["text"].values[0]
                line_text_no_tags = re.sub(r"<[^>]*>", "", str(line_text))
                evidence_texts.append(line_text_no_tags.strip())

            results.append(
                {
                    "note_id": note_id,
                    "predicted_code": code,
                    "predicted_code_probability": prob,
                    "evidence_line_numbers": evidence_lines,
                    "evidence_spans": evidence_spans,
                    "evidence_texts": evidence_texts,
                    "evidence_attributions": evidence_attributions,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    results_df.to_parquet(f"{output_filename}.parquet", index=False)

    LOGGER.info(
        "Processed %d notes with %d entities", df["note_id"].nunique(), len(df)
    )
    LOGGER.info("Generated %d ICD code predictions", len(results_df))
    LOGGER.info("Results written to %s", output_csv)
    LOGGER.info("Results written to %s.parquet", output_filename)


if __name__ == "__main__":
    main()
