"""Extract clinical entities from note files for the entity-only coding model.

This script is the first stage of the project pipeline. It cleans each clinical
note, runs the RoBERTa NER model, classifies assertion status for each detected
entity, and writes one row per retained entity. The per-entity rows are then
joined into the entity-only document fed to PLM-CA by
``ner/create_train_input.py`` (for training shards) or directly by the inference
scripts under ``external/plm_ca/``.

Input is a CSV or parquet file with ``note_id`` and ``text`` columns. Output is a
CSV with ``note_id,start_index,end_index,text``, where ``text`` is the entity
surface form wrapped in its type tag and the offsets index the cleaned note. The
optional visualisation flags also write cleaned note text and displaCy HTML
under ``results/``.

Example:
    python ner/extract_entities.py data/sample_data/sample_notes.csv --output_file results/ner/sample_entities.csv
"""

import argparse
import concurrent.futures
import csv
import logging
import multiprocessing
from pathlib import Path
import sys
import time
import threading

import pandas as pd
import torch
from spacy.tokens import Span
from spacy import displacy
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    RobertaConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

LOGGER = logging.getLogger("extract_entities")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FORMATTED_TEXTS_DIR = REPO_ROOT / "results" / "formatted_texts"
NER_DOCS_DIR = REPO_ROOT / "results" / "ner" / "docs_with_ner"
REQUIRED_INPUT_COLUMNS = ("note_id", "text")
OUTPUT_COLUMNS = ("note_id", "start_index", "end_index", "text")

# Ensure the ``ner`` package directory is importable when this file is
# invoked as a script (``python ner/extract_entities.py``). ProcessPoolExecutor
# workers inherit ``sys.path`` from the parent, so this also covers them.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Force UTF-8 on stdio so unicode prints render on Windows (cp1252) too.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from document_cleaning import (  # noqa: E402
    build_cleaning_nlp,
    clean_document_sentences,
)

# -----------------------------------------------------------------------------
# Model Loading & Configuration
# -----------------------------------------------------------------------------

# Global variables for model paths
NER_MODEL_DIR = REPO_ROOT / "data" / "models" / "ner_model"
AC_MODEL_DIR = REPO_ROOT / "data" / "models" / "ac_model"

# Thread-local storage for models to avoid reloading in each worker
thread_local_data = threading.local()


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def get_models():
    """Get or initialize models for the current thread/process."""
    if not hasattr(thread_local_data, 'models_loaded'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load NER model
        tokenizer_NER = RobertaTokenizerFast.from_pretrained(str(NER_MODEL_DIR), add_prefix_space=False)
        config_path = NER_MODEL_DIR / "model_config.json"
        config = RobertaConfig.from_json_file(str(config_path))
        model = RobertaForTokenClassification(config)
        model_weights_path = NER_MODEL_DIR / "final_model_for_inference.pt"
        model.load_state_dict(torch.load(str(model_weights_path), map_location=device))
        model.to(device)
        model.eval()
        
        # Load assertion classifier
        tokenizer_AC = AutoTokenizer.from_pretrained(str(AC_MODEL_DIR))
        model_AC = AutoModelForSequenceClassification.from_pretrained(str(AC_MODEL_DIR)).to(device)
        classifier = TextClassificationPipeline(
            model=model_AC, tokenizer=tokenizer_AC, device=0 if torch.cuda.is_available() else -1
        )
        
        thread_local_data.device = device
        thread_local_data.tokenizer_NER = tokenizer_NER
        thread_local_data.model = model
        thread_local_data.classifier = classifier
        thread_local_data.models_loaded = True
    
    return thread_local_data.device, thread_local_data.tokenizer_NER, thread_local_data.model, thread_local_data.classifier

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def infer_sentences(text: str) -> list:
    """
    Perform NER inference on the input text. Returns a list of entity dictionaries
    containing 'start', 'end', and 'label'.
    """
    device, tokenizer_NER, model, _ = get_models()
    
    inputs = tokenizer_NER(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)

    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    word_ids = inputs.word_ids()

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()[0]

    id_to_label = {
        0: "O",
        1: "B-procedure",
        2: "I-procedure",
        3: "B-health_context",
        4: "I-health_context",
        5: "B-disorder",
        6: "I-disorder",
        7: "B-normal_finding",
        8: "I-normal_finding",
        9: "B-abnormal_finding",
        10: "I-abnormal_finding",
        11: "B-medication",
        12: "I-medication"
    }
    pred_labels = [id_to_label[pred] for pred in predictions]

    entities = []
    current_entity = None
    for idx, (label, offset) in enumerate(zip(pred_labels, offset_mapping)):
        # Skip padding tokens and 'O' labels
        word_id = word_ids[idx]
        if word_id is None or label == 'O':
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        start_char, end_char = offset
        if label.startswith('B-'):
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {'start': start_char, 'end': end_char, 'label': label[2:]}
        elif label.startswith('I-'):
            if current_entity is not None:
                current_entity['end'] = end_char
            else:
                current_entity = {'start': start_char, 'end': end_char, 'label': label[2:]}
        else:
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = None

    if current_entity is not None:
        entities.append(current_entity)
    return entities

def get_token_length(text: str) -> int:
    """
    Returns the number of tokens in the text using the NER (RoBERTa) tokenizer.
    """
    _, tokenizer_NER, _, _ = get_models()
    return len(tokenizer_NER.encode(text))

def process_batch(batch_sentences: list, batch_sentence_starts: list) -> list:
    """
    Process a batch of sentences by joining them, performing NER inference, and adjusting
    entity character offsets to account for their original positions.
    """
    batch_text = '\n'.join(batch_sentences)
    entities = infer_sentences(batch_text)

    adjusted_entities = []
    cumulative_positions = [0]
    for s in batch_sentences[:-1]:
        cumulative_positions.append(cumulative_positions[-1] + len(s) + 1)  # +1 for newline

    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        # Find the sentence that contains the entity
        for i, start_pos in enumerate(cumulative_positions):
            end_pos = start_pos + len(batch_sentences[i])
            if start_pos <= entity_start < end_pos:
                offset = batch_sentence_starts[i] - start_pos
                entity['start'] += offset
                entity['end'] += offset
                adjusted_entities.append(entity)
                break
    return adjusted_entities

# Functions to handle long sentence splitting
# These make up for flaws in the PyRuSH sentence splitter

def find_split_index(text: str, midpoint: int, punctuation_list: list) -> int:
    """
    Search backwards from the midpoint for any punctuation character in punctuation_list.
    Returns the index after the punctuation if found; otherwise, returns -1.
    """
    for i in range(midpoint, 0, -1):
        if text[i] in punctuation_list:
            return i + 1
    return -1

def split_into_subsentences(text: str, start_char: int, max_token_length: int) -> list:
    """
    Recursively split a sentence into smaller sub-sentences that fit under max_token_length.
    Splitting is attempted first on ('.', '!'), then on ('*', ',', '#', '?'), and if no
    suitable punctuation is found, the sentence is split at its midpoint.
    
    Returns a list of tuples: (sub_sentence, sub_sentence_start_char).
    """
    if get_token_length(text) <= max_token_length:
        return [(text, start_char)]

    midpoint = len(text) // 2
    split_index = find_split_index(text, midpoint, ['.', '!'])
    if split_index == -1:
        split_index = find_split_index(text, midpoint, ['*', ',', '#', '?'])
    if split_index == -1:
        split_index = midpoint

    left_chunk = text[:split_index].strip()
    right_chunk = text[split_index:].strip()

    left_subs = split_into_subsentences(left_chunk, start_char, max_token_length)
    right_subs = []
    if right_chunk:
        right_subs = split_into_subsentences(right_chunk, start_char + split_index, max_token_length)

    return left_subs + right_subs

def infer_DC(sentences: list, sentence_starts: list) -> list:
    """
    Groups sentences into batches based on token length limits and performs NER
    inference on each batch. Handles long sentences by splitting them into smaller
    sub-sentences.
    
    Returns a list of entity dictionaries.
    """
    max_token_length = 500  # Allow some buffer
    current_token_length = 0
    current_batch_sentences = []
    current_batch_sentence_starts = []
    all_entities = []

    for sent_text, sent_start_char in zip(sentences, sentence_starts):
        chunk_token_length = get_token_length(sent_text)
        if chunk_token_length > max_token_length:
            # Split long sentence into sub-sentences
            subsentences = split_into_subsentences(sent_text, sent_start_char, max_token_length)
            for sub_sent, sub_start in subsentences:
                sub_len = get_token_length(sub_sent)
                if current_token_length + sub_len > max_token_length and current_batch_sentences:
                    entities = process_batch(current_batch_sentences, current_batch_sentence_starts)
                    all_entities.extend(entities)
                    current_batch_sentences = []
                    current_batch_sentence_starts = []
                    current_token_length = 0
                current_batch_sentences.append(sub_sent)
                current_batch_sentence_starts.append(sub_start)
                current_token_length += sub_len
        else:
            if current_token_length + chunk_token_length > max_token_length and current_batch_sentences:
                entities = process_batch(current_batch_sentences, current_batch_sentence_starts)
                all_entities.extend(entities)
                current_batch_sentences = []
                current_batch_sentence_starts = []
                current_token_length = 0
            current_batch_sentences.append(sent_text)
            current_batch_sentence_starts.append(sent_start_char)
            current_token_length += chunk_token_length

    if current_batch_sentences:
        entities = process_batch(current_batch_sentences, current_batch_sentence_starts)
        all_entities.extend(entities)

    return all_entities

def check_assertion(text: str, label: str, classifier) -> str:
    """
    Use the assertion classifier to classify the assertion status of an entity.
    Returns a modified label in the format '<label>_<assertion>'.
    """
    idx_to_label = {
        'LABEL_0': 'ABSENT',
        'LABEL_1': 'FAMILY',
        'LABEL_2': 'HYPOTHETICAL',
        'LABEL_3': 'POSSIBLE',
        'LABEL_4': 'PRESENT'
    }
    classification = classifier(text)
    assertion = idx_to_label[classification[0]['label']]
    return f"{label}_{assertion}"

def spans_overlap(span1: Span, span2: Span) -> bool:
    """
    Check if two spaCy spans overlap.
    """
    return span1.end > span2.start

# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------

def process_single_document(
    text: str,
    note_id,
    save_formatted: bool = False,
    save_ner_docs: bool = False,
    log_level: str = "INFO",
) -> tuple:
    """
    Process a single document:
      1. Clean and split the text into sentences.
      2. Adjust sentence boundaries based on section titles.
      3. Reconnect erroneously split sentences.
      4. Reassemble the document while tracking character offsets.
      5. Run NER inference and adjust entity spans.
      6. Update each entity's label using assertion classification.
      7. Filter out unwanted entities and sections.
      8. OPTIONALLY save the formatted text and NER results for visualisation.
    
    Returns:
      A tuple (note_id, index_data), where index_data is a list of [start_index, end_index, text]
      for the remaining entities.
    """
    configure_logging(log_level)

    # Initialize models for this worker process.
    _, _, _, classifier = get_models()

    # Build spaCy pipeline (PyRuSH + Sectionizer) for this worker. Each worker
    # gets its own instance because Sectionizer state is not shared across
    # processes.
    nlp = build_cleaning_nlp()

    # Clean the document using the shared pipeline.
    new_doc, cleaned_sentences, sentence_starts = clean_document_sentences(text, nlp=nlp)

    # Run NER inference on the cleaned sentences
    entities = infer_DC(cleaned_sentences, sentence_starts)

    # Load the reassembled text into spaCy
    doc = nlp(new_doc)

    # Optionally save the formatted text for visualization
    if save_formatted:
        FORMATTED_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
        formatted_filename = FORMATTED_TEXTS_DIR / f'formatted_{note_id}.txt'
        with open(formatted_filename, 'w', encoding='utf-8') as f:
            f.write(doc.text)

    valid_spans = []
    last_valid_span = None
    for ent in entities:
        span = doc.char_span(ent['start'], ent['end'], label=ent['label'], alignment_mode='expand')
        if span is None:
            LOGGER.warning("Span creation failed for entity at [%s, %s] in note %s", ent['start'], ent['end'], note_id)
            continue
        if last_valid_span is None:
            valid_spans.append(span)
            last_valid_span = span
        else:
            if spans_overlap(last_valid_span, span):
                LOGGER.warning("Span overlap in note %s at [%s, %s]", note_id, ent['start'], ent['end'])
                strict_span = doc.char_span(ent['start'], ent['end'], label=ent['label'], alignment_mode='strict')
                if strict_span is not None:
                    valid_spans.append(strict_span)
                    last_valid_span = strict_span
            else:
                valid_spans.append(span)
                last_valid_span = span

    doc.ents = valid_spans

    # Update entity labels using assertion classifier
    ents = list(doc.ents)
    for i, ent in enumerate(ents):
        sentence = ent.sent
        start_within_sentence = max(sentence.start, ent.start - 100)
        end_within_sentence = min(sentence.end, ent.end + 25)
        context_left = doc[start_within_sentence:ent.start].text
        context_right = doc[ent.end:end_within_sentence].text
        input_text = f"{context_left} <entity> {ent.text} <entity> {context_right}"
        result = check_assertion(input_text, ent.label_, classifier)
        new_ent = Span(doc, ent.start, ent.end, label=result)
        ents[i] = new_ent
    doc.ents = ents

    # Assertion filtering defines the entity-only document passed to PLM-CA, so
    # keep the default set aligned with the paper-era pipeline.
    excluded_labels = {
        "disorder_ABSENT",
        "disorder_HYPOTHETICAL",
        "abnormal_finding_ABSENT",
        "abnormal_finding_HYPOTHETICAL",
        "health_context_ABSENT",
        "health_context_HYPOTHETICAL",
        "health_context_FAMILY",
        "medication_ABSENT",
        "medication_HYPOTHETICAL",
        "medication_POSSIBLE",
        "medication_FAMILY",
        "procedure_ABSENT",
        "procedure_HYPOTHETICAL",
        "procedure_POSSIBLE",
        "procedure_FAMILY",
        "normal_finding_PRESENT",
        "normal_finding_HYPOTHETICAL",
        "normal_finding_POSSIBLE",
        "normal_finding_FAMILY",
        "normal_finding_ABSENT"
    }

    captured_sections = []
    index_data = []

    for ent in doc.ents:
        if ent.label_ in excluded_labels:
            continue
        if ent._.section and ent._.section.category == "medications_discharge":
            continue
        if ent._.section and ent._.section.category == 'allergies':
            start_idx = ent.start_char
            end_idx = ent.end_char
            text = "Allergy: " + ent.text.replace("\n", ' ')
            index_data.append([start_idx, end_idx, text])
            continue
        if ent._.section and ent._.section.category == "family_history":
            label_split = ent.label_.rsplit("_", 1)
            base_type = label_split[0]
            assertion = label_split[1] if len(label_split) > 1 else ""
            if base_type not in ["disorder", "abnormal_finding"] or assertion != "FAMILY":
                continue

        if ent._.section and ent._.section.category not in ["allergies", "medications_discharge"]:
            title_start, title_end = ent._.section.title_span
            section_text = doc[title_start:title_end].text
            if not captured_sections or section_text != captured_sections[-1]:
                captured_sections.append(section_text)
                index_data.append([doc[title_start].idx, doc[title_end - 1].idx, str(ent._.section.category)])

        start_idx = ent.start_char
        end_idx = ent.end_char
        label_split = ent.label_.rsplit("_", 1)
        base_type = label_split[0]
        assertion = label_split[1] if len(label_split) > 1 else ""
        extra_prefix = ""
        if assertion == "POSSIBLE":
            extra_prefix = "Possible: "
        elif assertion == "FAMILY":
            extra_prefix = "Family history: "
        ent_text = ent.text.replace('\n', ' ')
        text = f"<{base_type}> {extra_prefix}{ent_text}"
        index_data.append([start_idx, end_idx, text])

    # Save the note with entities highlighted in HTML format
    # If additional entity-assertion combinations are included above, add them
    # to the palette so the HTML output can render them consistently.
    if save_ner_docs:
        PALETTE = {
            "disorder": {
                "PRESENT": "#60A5FA",
                "POSSIBLE": "#93C5FD",
                "HYPOTHETICAL": "#BFDBFE",
                "ABSENT": "#DBEAFE",
                "FAMILY": "#EFF6FF",
            },
            "procedure": {
                "PRESENT": "#A78BFA",
                "POSSIBLE": "#C4B5FD",
                "HYPOTHETICAL": "#DDD6FE",
                "ABSENT": "#EDE9FE",
                "FAMILY": "#F5F3FF",
            },
            "medication": {
                "PRESENT": "#34D399",
                "POSSIBLE": "#6EE7B7",
                "HYPOTHETICAL": "#A7F3D0",
                "ABSENT": "#D1FAE5",
                "FAMILY": "#ECFDF5",
            },
            "abnormal_finding": {
                "PRESENT": "#FB923C",
                "POSSIBLE": "#FDBA74",
                "HYPOTHETICAL": "#FED7AA",
                "ABSENT": "#FFEDD5",
                "FAMILY": "#FFF7ED",
            },
            "normal_finding": {
                "PRESENT": "#2DD4BF",
                "POSSIBLE": "#5EEAD4",
                "HYPOTHETICAL": "#99F6E4",
                "ABSENT": "#CCFBF1",
                "FAMILY": "#F0FDF4",
            },
            "health_context": {
                "PRESENT": "#A3A3A3",
                "POSSIBLE": "#D4D4D4",
                "HYPOTHETICAL": "#E5E5E5",
                "ABSENT": "#F5F5F5",
                "FAMILY": "#FAFAFA",
            },
        }

        def build_colors(palette: dict) -> dict:
            out = {}
            for concept, shades in palette.items():
                for status, hexv in shades.items():
                    out[f"{concept}_{status}"] = hexv
            return out

        STATUS_ORDER = ["PRESENT", "POSSIBLE", "HYPOTHETICAL", "ABSENT", "FAMILY"]
        CONCEPT_ORDER = [
            "disorder",
            "procedure",
            "medication",
            "abnormal_finding",
            "normal_finding",
            "health_context",
        ]

        # Final colors dict and ordered ents list
        colors = build_colors(PALETTE)
        ents = [f"{c}_{s}" for c in CONCEPT_ORDER for s in STATUS_ORDER]

        options = {"ents": ents, "colors": colors}

        html = displacy.render(doc, style="ent", jupyter=False, options=options)
        html = html.replace(".entity {", ".entity {margin-bottom: 5px; line-height: 3;")
        NER_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        with open(NER_DOCS_DIR / f"NER_{note_id}.html", "w", encoding="utf-8") as file:
            file.write(html)

    return note_id, index_data

# -----------------------------------------------------------------------------
# Main Parallel Processing Function
# -----------------------------------------------------------------------------

def validate_input_file(filename: str):
    """Read a note CSV/parquet after validating the required input schema."""
    input_path = Path(filename)
    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        LOGGER.error("Make sure the file exists and the path is correct.")
        return None
    
    try:
        file_ext = input_path.suffix.lower()
        
        if file_ext == '.parquet':
            docs = pd.read_parquet(input_path)
            file_type = "parquet"
        elif file_ext == '.csv':
            docs = pd.read_csv(input_path)
            file_type = "CSV"
        else:
            LOGGER.error("Unsupported file format: %s", file_ext)
            LOGGER.error("Supported formats: .parquet, .csv")
            return None
        
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in docs.columns]
        if missing:
            LOGGER.error("Required columns missing from input %s: %s", input_path, missing)
            LOGGER.error("Found columns: %s", list(docs.columns))
            return None
        
        LOGGER.info("Input %s file validated: %d notes ready for processing", file_type, len(docs))
        return docs
    except Exception as e:
        LOGGER.error("Error reading input file: %s", e)
        return None

def validate_model_files():
    """Validate that all required model files exist."""
    ner_files = [
        (NER_MODEL_DIR / "final_model_for_inference.pt", "NER model weights"),
        (NER_MODEL_DIR / "model_config.json", "NER model config"),
        (AC_MODEL_DIR, "Assertion Classification model directory"),
    ]
    
    for file_path, description in ner_files:
        if not file_path.exists():
            LOGGER.error("%s not found at %s", description, file_path)
            LOGGER.error("Make sure you've downloaded all models with: python data_download.py")
            return False
    
    LOGGER.info("All required model files found")
    return True

def process_documents_from_csv_parallel(
    filename: str,
    output_filename: str = 'results/ner/output_entities.csv',
    max_workers: int = 2,
    save_formatted: bool = False,
    save_ner_docs: bool = False,
    force: bool = False,
    log_level: str = "INFO",
):
    """
    Process documents from a CSV/parquet file in parallel.
    The input file must contain ``note_id`` and ``text`` columns.
    Outputs the extracted entities to a CSV file.

    If ``force`` is True, any existing output file is truncated and every
    document is reprocessed; otherwise note_ids already present in the
    output CSV are skipped (resume-from-partial behaviour).
    """
    # Validate input and models
    docs = validate_input_file(filename)
    if docs is None:
        return False

    if not validate_model_files():
        return False

    LOGGER.info("Using %d parallel workers", max_workers)
    LOGGER.info("GPU available: %s", torch.cuda.is_available())

    output_path = Path(output_filename)
    if output_path.parent != Path(".") and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Created output directory: %s", output_path.parent)

    processed_note_ids = set()

    if force and output_path.exists():
        output_path.unlink()
        LOGGER.info("--force: removed existing output, reprocessing all documents.")
    elif output_path.exists() and output_path.stat().st_size > 0:
        df_processed = pd.read_csv(output_path)
        if "note_id" not in df_processed.columns:
            LOGGER.error("Existing output %s is missing required column: note_id", output_path)
            return False
        processed_note_ids.update(df_processed['note_id'].unique())
        LOGGER.info(
            "Resuming processing. %d documents already processed (pass --force to reprocess).",
            len(processed_note_ids),
        )
    else:
        LOGGER.info("Starting fresh processing.")

    # Set multiprocessing start method to 'spawn' for CUDA compatibility across OS
    mp_context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        futures = []
        for row in docs.itertuples():
            note_id = row.note_id
            if note_id not in processed_note_ids:
                future = executor.submit(
                    process_single_document,
                    row.text,
                    note_id,
                    save_formatted,
                    save_ner_docs,
                    log_level,
                )
                futures.append(future)

        processed = len(processed_note_ids)
        with output_path.open('a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            if output_path.stat().st_size == 0:
                csvwriter.writerow(OUTPUT_COLUMNS)

            for future in concurrent.futures.as_completed(futures):
                try:
                    note_id, index_data = future.result()
                    for row in index_data:
                        start_index, end_index, text = row
                        csvwriter.writerow([note_id, start_index, end_index, text])
                    processed += 1
                    csvfile.flush()
                    LOGGER.info("Processed document: %d (Note ID: %s)", processed, note_id)
                except Exception as e:
                    LOGGER.error(
                        "Error processing document: %s",
                        e,
                        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                    )
                    return False
    return True

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a parquet or CSV file to extract NER entities using custom pipelines."
    )
    parser.add_argument(
        "input_file", help="Path to the input file (.parquet or .csv, must contain 'text' and 'note_id' columns)."
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        dest="output_file",
        type=str,
        default="results/ner/output_entities.csv",
        help="Output filepath for the results CSV file (default: results/ner/output_entities.csv)."
    )
    parser.add_argument(
        "--max_workers",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)."
    )
    parser.add_argument(
        "--save-formatted-texts",
        "--save_formatted_texts",
        dest="save_formatted_texts",
        action="store_true",
        help="Save formatted text files for visualization (stored in results/formatted_texts/)."
    )
    parser.add_argument(
        "--save-ner-docs",
        "--save_ner_docs",
        dest="save_ner_docs",
        action="store_true",
        help="Save NER processed documents as HTML files in results/ner/docs_with_ner/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the resume-from-existing-output check: delete any existing output file and reprocess every document from scratch."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main():
    """Main function to handle command line arguments and run entity extraction."""
    args = parse_args()
    configure_logging(args.log_level)

    if args.save_formatted_texts:
        LOGGER.info("Formatted text saving enabled for visualization")

    if args.save_ner_docs:
        LOGGER.info("Saving NER processed documents as HTML enabled")

    LOGGER.info("Starting entity extraction")
    start_time = time.time()
    success = process_documents_from_csv_parallel(
        args.input_file,
        output_filename=args.output_file,
        max_workers=args.max_workers,
        save_formatted=args.save_formatted_texts,
        save_ner_docs=args.save_ner_docs,
        force=args.force,
        log_level=args.log_level,
    )
    end_time = time.time()
    if not success:
        sys.exit(1)
    
    # Log completion summary
    output_path = Path(args.output_file)
    if output_path.exists():
        try:
            df_results = pd.read_csv(output_path)
            LOGGER.info("Entity extraction completed successfully")
            LOGGER.info("Processing time: %.1f seconds", end_time - start_time)
            LOGGER.info(
                "Extracted %d entities from %d notes",
                len(df_results),
                df_results['note_id'].nunique(),
            )
            LOGGER.info("Results saved to: %s", output_path)
            LOGGER.info("Next step: Run ICD coding inference")
            LOGGER.info("  cd external/plm_ca")
            entities_hint = output_path if output_path.is_absolute() else Path("..") / ".." / output_path
            LOGGER.info(
                "  python infer_with_explanations.py %s ../../results/coded/filename",
                entities_hint,
            )
        except Exception as e:
            LOGGER.warning("Processing completed but could not read output file: %s", e)
    else:
        LOGGER.warning("Processing completed but output file not found: %s", output_path)


if __name__ == '__main__':
    main()
