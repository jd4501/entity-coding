"""Predict ICD-10 codes from full-text notes and extract per-word evidence.

Full-text counterpart to infer_with_explanations.py. Both arms run PLM-CA
(the AttInGrad-explainable PLM-ICD variant from Edin et al., 2024); the
checkpoint loaded here was trained on raw MIMIC-IV discharge summaries via
train_plm_fulltext.py and serves as the project's full-text comparator
against the entity-only checkpoint. The two outputs feed
code_evidence/code_evidence_eval.ipynb, which compares both arms' evidence
against the MDACE ground-truth code-evidence spans (manuscript Tables 8 and
12).

This script handles only the first half of the full-text evidence extraction
on its own; the merge stage runs separately. The split is deliberate:

  1. (this script) Per-word AttInGrad extraction. Run at a permissive
     attribution threshold (the default ``1e-6`` retains essentially every
     word). Saving every word means the threshold can be re-tuned later
     without re-running inference (the inference is the slow step).
  2. merge_contiguous_spans.py: apply a final, val-tuned attribution
     threshold to drop low-signal words, then collapse the survivors into
     contiguous character-level spans (the unit MDACE annotates and that
     the notebook evaluates against).

End-to-end full-text evidence-extraction flow. Roughly: build MDACE-DC ->
extract per-word -> tune the threshold on val -> merge train and test ->
evaluate. In detail:

  a. Acquire MIMIC-III, MIMIC-IV, MIMIC-IV-Note, and the MDACE annotations
     into external/plm_ca/data/raw/ per external/plm_ca/README.md (MDACE
     itself is vendored under data/raw/MDace/). Then from external/plm_ca/,
     run the MIMIC-III, MIMIC-IV, MDACE preparation modules followed by
     ``make_mdace_icd10_inpatient``. This builds
     data/processed/mdace_icd10_inpatient/{train,val,test}.parquet.
     Each row carries the MIMIC-III discharge-note ``text`` joined in from
     NOTEEVENTS.csv.gz plus the MDACE code annotations.
  b. Run Stage 1 (Preprocessing) of code_evidence/code_evidence_eval.ipynb
     to filter to the discharge-summary-only subset used by the paper. That
     emits mdace_{val,train,test}_DConly.parquet ("DC" = discharge note
     only); uncomment the .to_csv lines in that cell for the CSV form this
     script reads. Run the cell from the repo root so the files land beside
     README.md, where the ``../../<name>`` invocation pattern below expects
     to find them when this script is launched from external/plm_ca/.
  c. Run this script over *all three* splits (val, train, test) at the
     default ``attr_threshold`` of ``1e-6``. The val output is needed for
     threshold tuning in step (d); the train and test outputs are needed
     for the merge in step (e).
  d. In Stage 2 (Threshold Tuning) of the notebook, sweep attribution
     thresholds on the *val* per-word output and pick the one that maximises
     partial-match F2 against MDACE ground-truth evidence spans. Val acts as
     the held-out tuning split here - it is *not* the model's training
     split (that was MIMIC-IV via train_plm_fulltext.py), it is held out
     specifically for the evidence-extraction hyperparameter. The paper's
     value was 0.0013877551020408164 (partial F2 = 0.4079 on val).
  e. Pass that val-derived threshold to merge_contiguous_spans.py when
     merging the train and test per-word outputs into contiguous spans.
  f. Run Stage 4 (Overall Classification Metrics on MDACE) on the
     concatenated (train + test) merged output for the final full-text
     evidence metrics. Train + test (rather than test alone) follows the
     manuscript's protocol: val is consumed by tuning, so the
     evidence-overlap evaluation uses the remaining MDACE-DC notes.

Usage (from external/plm_ca/, with the entitycoding conda env active):

    python infer_with_explanations_fulltext.py <input>[.csv] [output_basename] [attr_threshold]

``<input>`` is a CSV with columns ``note_id`` and ``text`` (one row per note).
``<input>`` and ``[output_basename]`` may include a path prefix; the canonical
layout places the DConly CSVs and fft_ ("full-text") outputs at the repo
root, so invocations from external/plm_ca/ look like
``../../mdace_val_DConly ../../fft_inferred_notes_with_evidence_mdace_val_DConly``.
The full-text PLM-CA checkpoint is the single copy trained once for this
project and downloadable via ``data_download.py --models fulltext`` (lands
under models/fulltext/).

Output schema (one row per predicted (note_id, code), parallel lists per
row. Consumed by merge_contiguous_spans.py and the notebook):
  * note_id
  * predicted_code
  * predicted_code_probability:    ``PROBABILITY_THRESHOLD`` in the
                                   notebook must match --decision-boundary
                                   here. The notebook re-applies the
                                   threshold when computing metrics; if the
                                   two disagree, the notebook will filter
                                   out predictions this script chose to
                                   keep (or vice versa) and the evaluation
                                   set will silently shift.
  * evidence_texts:                list[str], one entry per word
  * evidence_spans:                list[tuple[int, int]] of (start, end)
                                   char offsets
  * evidence_attributions:         list[float], parallel to evidence_texts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

LOGGER = logging.getLogger("infer_with_explanations_fulltext")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
FULLTEXT_MODEL_DIR = SCRIPT_DIR / "models" / "fulltext"

# Tokenization budget. Matches the full-text model's training-time max_length
# (see models/fulltext/config.yaml: data.max_length and
# model.configs.chunk_size).
MAX_LENGTH = 6000

# Default decision boundary used to produce the paper's full-text results.
# Mirrors PROBABILITY_THRESHOLD in code_evidence/code_evidence_eval.ipynb when
# --decision-boundary is left unset; see docs/reproduce.md#tuned-thresholds.
DEFAULT_DECISION_BOUNDARY = 0.4141414165496826

# Permissive default attribution cutoff: effectively zero, retains every
# per-word attribution so downstream val tuning has the full distribution to
# sweep over.
DEFAULT_ATTR_THRESHOLD = 1e-6

REQUIRED_INPUT_COLUMNS = ("note_id", "text")


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
            "Full-text PLM-CA inference with AttInGrad per-word evidence "
            "extraction. See module docstring for the end-to-end MDACE flow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (from external/plm_ca/):\n"
            "  python infer_with_explanations_fulltext.py \\\n"
            "      ../../mdace_val_DConly \\\n"
            "      ../../fft_inferred_notes_with_evidence_mdace_val_DConly 1e-6"
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Input CSV with columns note_id, text. "
            "May be passed with or without the .csv suffix."
        ),
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=(
            "Output prefix (no suffix; both .csv and .parquet are written). "
            "Default when omitted: ../fft_inferred_notes_with_evidence_<input-basename>. "
            "Pass an explicit ../../... prefix to write beside the repo README."
        ),
    )
    parser.add_argument(
        "attr_threshold",
        nargs="?",
        type=float,
        default=DEFAULT_ATTR_THRESHOLD,
        help=(
            "Per-word attribution cutoff. Use 1e-6 (default, near-zero) when "
            "extracting for downstream val-tuning; pass a tuned value at "
            "merge-time. Words with abs(attr) below this are dropped, with a "
            "fallback to the single highest-attr word per code."
        ),
    )
    parser.add_argument(
        "--decision-boundary",
        type=float,
        default=DEFAULT_DECISION_BOUNDARY,
        help=(
            "Probability threshold above which a code is predicted. Default is "
            "the paper-tuned value for the full-text model and matches "
            "PROBABILITY_THRESHOLD in code_evidence/code_evidence_eval.ipynb."
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
    """Read the input CSV and confirm it has note_id and text columns."""
    if not filepath.exists():
        LOGGER.error("Input file not found: %s", filepath)
        LOGGER.error(
            "Stage 1 of code_evidence/code_evidence_eval.ipynb produces the "
            "MDACE-DC parquets; uncomment the .to_csv lines there to emit the "
            "CSV form this script reads."
        )
        sys.exit(1)

    df = pd.read_csv(filepath)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.error("Input file missing required columns: %s", missing)
        LOGGER.error("Found columns: %s", list(df.columns))
        sys.exit(1)

    LOGGER.info("Input validated: %d notes from %s", len(df), filepath)
    return df


def validate_model_files() -> None:
    """Confirm the full-text checkpoint is present."""
    required = [
        (FULLTEXT_MODEL_DIR / "target_tokenizer.json", "ICD code mapping"),
        (FULLTEXT_MODEL_DIR / "config.yaml", "Model configuration"),
        (FULLTEXT_MODEL_DIR / "best_model.pt", "Trained model weights"),
    ]
    for path, description in required:
        if not path.exists():
            LOGGER.error("%s not found at %s", description, path)
            LOGGER.error(
                "Run `python data_download.py --models fulltext` from the repo "
                "root to fetch the full-text checkpoint."
            )
            sys.exit(1)
    LOGGER.debug("All required model files found")


def _extract_word_level_evidence(
    full_text: str,
    word_ids: list,
    offset_mapping,
    attr_vector,
    attr_threshold: float,
) -> list[dict]:
    """Aggregate subword attributions up to whole words, then filter.

    The model's RoBERTa-based subword tokenizer can split a single word into
    multiple subword tokens, and AttInGrad emits an attribution per subtoken.
    To report
    evidence at the word level (the unit consumed by
    merge_contiguous_spans.py and the MDACE evaluation), each word's
    subtokens are combined under a deliberate aggregation:

      - attribution: raw ``max`` over subtokens (not sum, not abs-max). Sum
        would over-reward longer words; abs-max would treat a strong negative
        contribution as positive evidence.
      - char span: ``(min(subword_starts), max(subword_ends))``, so the
        word's offsets cover its full extent in the source text.

    Words with ``abs(attribution) < attr_threshold`` are dropped. If that
    removes every word for the current (note_id, code) pair, the single word
    with the largest absolute attribution is kept; the same >=1-evidence
    fallback merge_contiguous_spans.py applies after threshold-tuning.
    """
    word_map: dict[int, dict] = {}
    for subword_idx, w_id in enumerate(word_ids):
        if w_id is None:  # special tokens carry no source-text offset
            continue
        subword_attr = attr_vector[subword_idx].item()
        start_char, end_char = offset_mapping[subword_idx]
        if w_id not in word_map:
            word_map[w_id] = {
                "start": int(start_char),
                "end": int(end_char),
                "attribution": subword_attr,
            }
        else:
            w = word_map[w_id]
            w["start"] = min(w["start"], int(start_char))
            w["end"] = max(w["end"], int(end_char))
            w["attribution"] = max(w["attribution"], subword_attr)

    word_level_data = [
        {
            "word": full_text[info["start"] : info["end"]],
            "start": info["start"],
            "end": info["end"],
            "attribution": info["attribution"],
        }
        for info in word_map.values()
    ]

    filtered = [w for w in word_level_data if abs(w["attribution"]) >= attr_threshold]
    if not filtered and word_level_data:
        filtered = [max(word_level_data, key=lambda w: abs(w["attribution"]))]

    filtered.sort(key=lambda w: abs(w["attribution"]), reverse=True)
    return filtered


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    filename = _strip_known_suffix(args.input_file)
    if args.output_file is not None:
        output_filename = _strip_known_suffix(args.output_file)
    else:
        output_filename = (
            f"../fft_inferred_notes_with_evidence_{os.path.basename(filename)}"
        )

    input_filepath = Path(f"{filename}.csv")
    df = validate_input_file(input_filepath)

    output_csv = Path(f"{output_filename}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    validate_model_files()

    LOGGER.info("Loading full-text PLM-CA model")
    with open(FULLTEXT_MODEL_DIR / "target_tokenizer.json", "r") as f:
        codes = json.load(f)

    saved_config = OmegaConf.load(FULLTEXT_MODEL_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    encoder_path = _resolve_encoder_path(saved_config.model.configs.model_path)
    saved_config.model.configs.model_path = encoder_path
    text_tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    model, _ckpt_boundary = load_trained_model(
        FULLTEXT_MODEL_DIR,
        saved_config,
        pad_token_id=text_tokenizer.pad_token_id,
        device=device,
    )
    model.eval()
    model.to(device)

    explainer = get_grad_attention_callable(model)

    decision_boundary = args.decision_boundary
    attr_threshold = args.attr_threshold

    show_progress = LOGGER.isEnabledFor(logging.INFO)
    iterator = (
        track(df.iterrows(), total=len(df), description="Processing notes")
        if show_progress
        else df.iterrows()
    )

    results: list[dict] = []
    for _, row in iterator:
        note_id = row["note_id"]
        full_text = str(row["text"])

        encoded = text_tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offset_mapping = encoded["offset_mapping"][0].cpu().numpy()
        word_ids = encoded.word_ids(batch_index=0)
        if word_ids is None:
            LOGGER.warning(
                "word_ids() unavailable for note %s; skipping per-word aggregation.",
                note_id,
            )
            continue

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)

        predicted_labels = (probs > decision_boundary).nonzero(as_tuple=False)
        if len(predicted_labels) == 0:
            continue

        target_ids = predicted_labels[:, 1]
        attributions = explainer(input_ids, target_ids, device)
        predicted_probs = probs[0, target_ids].cpu().numpy()

        triplets = sorted(
            (
                (target_ids[j].item(), predicted_probs[j], attributions[:, j])
                for j in range(len(target_ids))
            ),
            key=lambda t: t[1],
            reverse=True,
        )

        for target_id, prob, attr_vector in triplets:
            code = codes[target_id]
            words = _extract_word_level_evidence(
                full_text, word_ids, offset_mapping, attr_vector, attr_threshold
            )
            if not words:
                continue

            results.append(
                {
                    "note_id": note_id,
                    "predicted_code": code,
                    "predicted_code_probability": float(prob),
                    "evidence_texts": [w["word"] for w in words],
                    "evidence_spans": [(w["start"], w["end"]) for w in words],
                    "evidence_attributions": [w["attribution"] for w in words],
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    results_df.to_parquet(f"{output_filename}.parquet", index=False)

    LOGGER.info("Processed %d notes", len(df))
    LOGGER.info("Generated %d (note_id, code) predictions", len(results_df))
    LOGGER.info("Per-word evidence written to %s", output_csv)
    LOGGER.info("Per-word evidence written to %s.parquet", output_filename)
    LOGGER.info(
        "Next: tune the attribution threshold on the val output via Stage 2 of "
        "code_evidence/code_evidence_eval.ipynb, then merge train/test with "
        "merge_contiguous_spans.py."
    )


if __name__ == "__main__":
    main()
