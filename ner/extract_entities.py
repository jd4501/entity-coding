import os
import re
import time
import csv
import torch
import spacy
import numpy as np
import pandas as pd
import medspacy
import concurrent.futures
import argparse
from functools import partial
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy import displacy
from concurrent.futures import ProcessPoolExecutor
from medspacy.section_detection import Sectionizer
from medspacy.sentence_splitting import PyRuSHSentencizer
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    RobertaConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# -----------------------------------------------------------------------------
# Model Loading & Configuration
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NER model from the saved directory
save_dir = "data/models/ner_model"
tokenizer_NER = RobertaTokenizerFast.from_pretrained(save_dir, add_prefix_space=False)
config_path = os.path.join(save_dir, "model_config.json")
config = RobertaConfig.from_json_file(config_path)
model = RobertaForTokenClassification(config)
model_weights_path = os.path.join(save_dir, "final_model_for_inference.pt")
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# Load assertion classifier
tokenizer_AC = AutoTokenizer.from_pretrained("data/models/ac_model")
model_AC = AutoModelForSequenceClassification.from_pretrained(
    "data/models/ac_model"
).to(device)
classifier = TextClassificationPipeline(
    model=model_AC, tokenizer=tokenizer_AC, device=0 if torch.cuda.is_available() else -1
)

# Toggle to save re-formatted notes for later visualization
save_formatted_texts = False

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by replacing consecutive whitespace with a single space.
    """
    return ' '.join(text.split())

def infer_sentences(text: str) -> list:
    """
    Perform NER inference on the input text. Returns a list of entity dictionaries
    containing 'start', 'end', and 'label'.
    """
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
    Returns the number of tokens in the text using the NER tokenizer.
    """
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

def check_assertion(text: str, label: str) -> str:
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

def reconnect_stop_word_splits(sentences: list, stop_words: set = None) -> list:
    """
    Merge adjacent sentences when a sentence ends with a stop word (e.g., articles,
    determiners, conjunctions) and the following sentence starts with a capital letter.
    
    :param sentences: List of sentence strings.
    :param stop_words: Set of lowercased stop words. If None, a default set is used.
    :return: A new list of merged sentences.
    """
    if stop_words is None:
        stop_words = {
            # Articles and determiners
            "a", "an", "the", "my", "your", "his", "her", "its", "their", "our", "one's",
            "this", "that", "these", "those", "some", "any", "enough", "many", "much",
            "few", "several", "all", "both", "each", "every", "either", "neither", "none",
            "plenty", "most",
            # Prepositions
            "in", "on", "at", "by", "to", "of", "for", "with", "from", "about", "like",
            "through", "over", "under", "between", "before", "after", "around", "near",
            "inside", "outside", "within", "without", "during", "against",
            # Conjunctions
            "and", "or", "but", "nor", "yet", "so",
            # Subordinating conjunctions
            "if", "since", "because", "although", "though", "when", "while", "until",
            "where", "unless", "as", "than", "once", "whether",
            # Modal/Auxiliary verbs
            "can", "could", "would", "should", "must", "may", "might",
            "is", "are", "was", "were", "has", "have", "had", "do", "does", "did",
            "will", "shall", "been",
            # Pronouns
            "he", "she", "it", "we", "you", "they", "one", "who", "whom",
            "whose", "what", "which", "someone", "something", "anyone",
            "anything", "everyone", "everything",
            # Adverbs
            "here", "there", "then", "so", "also", "too", "only", "still",
            "just", "even", "yet", "well",
            # Common contractions
            "it's", "he's", "she's", "they're", "we're", "you're", "I've",
            "we've", "they've", "he'd", "she'd", "they'd"
        }

    merged_sentences = []
    i = 0
    while i < len(sentences):
        current_sent = sentences[i].strip()
        if i < len(sentences) - 1:
            next_sent = sentences[i + 1].strip()
            last_word = current_sent.split()[-1].lower() if current_sent else ""
            first_char = next_sent[0] if next_sent else ""
            if last_word in stop_words and first_char.isupper():
                merged_sentences.append(current_sent + " " + next_sent)
                i += 2
                continue
        merged_sentences.append(current_sent)
        i += 1
    return merged_sentences

def spans_overlap(span1: Span, span2: Span) -> bool:
    """
    Check if two spaCy spans overlap.
    """
    return span1.end > span2.start

# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------

def process_single_document(text: str, note_id, pattern_midline_newlines, nlp, classifier, infer_DC) -> tuple:
    """
    Process a single document:
      1. Clean and split the text into sentences.
      2. Adjust sentence boundaries based on section titles.
      3. Reconnect erroneously split sentences.
      4. Reassemble the document while tracking character offsets.
      5. Run NER inference and adjust entity spans.
      6. Update each entity's label using assertion classification.
      7. Filter out unwanted entities and sections.
    
    Returns:
      A tuple (note_id, index_data), where index_data is a list of [start_index, end_index, text]
      for the remaining entities.
    """
    text_proc = text.strip()
    doc_sentence_splitted = nlp(text_proc)
    sections = doc_sentence_splitted._.sections

    # Split sentences and further split on mid-sentence section titles
    raw_sentences = []
    for sent in doc_sentence_splitted.sents:
        sent_start = sent.start
        sent_end = sent.end
        splits = [sent_start]
        for section in sections:
            title_start, title_end = section.title_span
            if sent_start < title_start < sent_end:
                splits.append(title_start)
        splits.append(sent_end)
        splits = sorted(set(splits))
        for i in range(len(splits) - 1):
            span = doc_sentence_splitted[splits[i]:splits[i + 1]]
            target = pattern_midline_newlines.sub(' ', span.text).strip()
            target = normalize_whitespace(target)
            if target:
                raw_sentences.append(target)

    # Reconnect sentences split at stop words
    fixed_sentences = reconnect_stop_word_splits(raw_sentences)

    # Reassemble document and track sentence start offsets
    cleaned_sentences = []
    sentence_starts = []
    current_pos = 0
    new_doc = ''
    for sent in fixed_sentences:
        cleaned_sentences.append(sent)
        sentence_starts.append(current_pos)
        current_pos += len(sent) + 1  # +1 for the newline separator
        new_doc += ('\n' if new_doc else '') + sent

    # Run NER inference on the cleaned sentences
    entities = infer_DC(cleaned_sentences, sentence_starts)

    # Load the reassembled text into spaCy
    doc = nlp(new_doc)

    # Optionally save the formatted text for visualization
    if save_formatted_texts:
        os.makedirs('results/formatted_texts', exist_ok=True)
        formatted_filename = os.path.join('results/formatted_texts', f'formatted_{note_id}.txt')
        with open(formatted_filename, 'w', encoding='utf-8') as f:
            f.write(doc.text)

    valid_spans = []
    last_valid_span = None
    for ent in entities:
        span = doc.char_span(ent['start'], ent['end'], label=ent['label'], alignment_mode='expand')
        if span is None:
            print('Span creation failed')
            continue
        if last_valid_span is None:
            valid_spans.append(span)
            last_valid_span = span
        else:
            if spans_overlap(last_valid_span, span):
                print(f'Span overlap in {note_id}')
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
        result = check_assertion(input_text, ent.label_)
        new_ent = Span(doc, ent.start, ent.end, label=result)
        ents[i] = new_ent
    doc.ents = ents

    # Filtering and output construction
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

    sections = doc._.sections
    output = []
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
            output.append(text)
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
                output.append(str(ent._.section.category))
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
        output.append(text)
        index_data.append([start_idx, end_idx, text])
        final_text = '\n'.join(output)

    return note_id, index_data

# -----------------------------------------------------------------------------
# Main Parallel Processing Function
# -----------------------------------------------------------------------------

pattern_midline_newlines = re.compile(r'(?<=\S|\s)\n(?=\S)')
nlp = spacy.blank("en")
# nlp.add_pipe("medspacy_pyrush")  # Uncomment to add the default sentence splitter

def process_documents_from_csv_parallel(filename: str, output_filename: str = 'ner/output_entities.csv', max_workers: int = 2):
    """
    Process documents from a parquet file in parallel.
    The input file must contain 'text' and 'note_id' columns.
    Outputs the extracted entities to a CSV file.
    """
    docs = pd.read_parquet(filename)
    if 'text' not in docs.columns or 'note_id' not in docs.columns:
        print("Invalid file: Required columns 'text' and 'note_id' not found.")
        return

    processed_note_ids = set()

    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        df_processed = pd.read_csv(output_filename)
        processed_note_ids.update(df_processed['note_id'].unique())
        print(f"Resuming processing. {len(processed_note_ids)} documents already processed.")
    else:
        print("Starting fresh processing.")

    # Reinitialize the pattern and nlp object for this function scope
    pattern_midline_newlines = re.compile(r'(?<=\S|\s)\n(?=\S)')
    nlp = spacy.blank("en")
    # nlp.add_pipe("medspacy_pyrush")

    @Language.factory("custom_splitter")
    def create_custom_splitter(nlp, name):
        return PyRuSHSentencizer(nlp=nlp, rules_path='ner/rush_rules.tsv')
    nlp.add_pipe("custom_splitter")

    @Language.factory("custom_sectionizer")
    def create_custom_sectionizer(nlp, name):
        return Sectionizer(nlp, rules='ner/section_patterns.json')
    nlp.add_pipe("custom_sectionizer", last=True)

    process_func = partial(
        process_single_document,
        pattern_midline_newlines=pattern_midline_newlines,
        nlp=nlp,
        classifier=classifier,
        infer_DC=infer_DC,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for row in docs.itertuples():
            note_id = row.note_id
            if note_id not in processed_note_ids:
                future = executor.submit(process_func, row.text, note_id)
                futures.append(future)

        processed = len(processed_note_ids)
        with open(output_filename, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            if os.path.getsize(output_filename) == 0:
                csvwriter.writerow(['note_id', 'start_index', 'end_index', 'text'])

            for future in concurrent.futures.as_completed(futures):
                try:
                    note_id, index_data = future.result()
                    for row in index_data:
                        start_index, end_index, text = row
                        csvwriter.writerow([note_id, start_index, end_index, text])
                    processed += 1
                    csvfile.flush()
                    print(f"Processed document: {processed} (Note ID: {note_id})")
                except Exception as e:
                    print(f"Error processing document: {e}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process a parquet file to extract NER entities using custom pipelines."
    )
    parser.add_argument(
        "input_file", help="Path to the input parquet file (must contain 'text' and 'note_id' columns)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ner/output_entities.csv",
        help="Output filepath for the results CSV file (default: ner/output_entities.csv)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)."
    )
    args = parser.parse_args()

    start_time = time.time()
    process_documents_from_csv_parallel(args.input_file, output_filename=args.output_file, max_workers=args.max_workers)
    end_time = time.time()
    print(f"The function took {end_time - start_time} seconds to run (note: this includes time to setup parallel processing (a few seconds).")