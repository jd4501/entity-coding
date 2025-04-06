import sys
import warnings
import re
import json
import os
import torch
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from rich.progress import track
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", message="Special tokens have been added")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from explainable_medical_coding.utils.loaders import load_trained_model
from explainable_medical_coding.explainability.explanation_methods import get_grad_attention_callable

# Read post-NER file from command line
if len(sys.argv) > 1:
    filename = sys.argv[1].rstrip('.csv')  # Strip .csv if provided
else:
    filename = "../ner/sample_notes_entities"

# Read output filename from command line (optional)
if len(sys.argv) > 2:
    output_filename = sys.argv[2].rstrip('.csv').rstrip('.parquet')  # Strip if provided
else:
    name = os.path.basename(filename)
    output_filename = f"../inferred_notes_with_evidence_{name}"

df = pd.read_csv(f"{filename}.csv")


# Load entity model
model_path = Path('models/entityonly')
target_codes_path = model_path / 'target_tokenizer.json'

# Load the index-to-code mapping used by the model
with open(target_codes_path, 'r') as file:
    codes = json.load(file)  

saved_config = OmegaConf.load(model_path / "config.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the text tokenizer, modified for the new special tokens
text_tokenizer = AutoTokenizer.from_pretrained('models/tokenizer_latest')

model, decision_boundary = load_trained_model(
    model_path,
    saved_config,
    pad_token_id=text_tokenizer.pad_token_id,
    device=device,
)

model.eval()
model.to(device)

if 'diagnosis_codes' in df.columns:
    df['diagnosis_codes'] = df['diagnosis_codes'].apply(lambda x: x.split(','))

grouped = df.groupby('note_id')
# Initialize the explainer (this is AttInGrad)
# Note: if you look in explainability/explanation_methods.py, 
# you'll find other methods for getting explanations
explainer = get_grad_attention_callable(model)

results = []
decision_boundary = 0.4040403962135315 # This is the tuned boundary for this model, but feel free to raise/lower it

for note_id, group in track(grouped, description="Processing notes"):
    group_sorted = group.sort_values('start_index').reset_index(drop=True)
    group_sorted['line_number'] = group_sorted.index + 1 # Line numbers starting from 1

    lines = group_sorted['text'].tolist()
    line_spans = []
    pos = 0
    lines_with_newlines = []

    for line in lines:
        line = str(line)
        start_pos = pos
        end_pos = pos + len(line)
        line_spans.append((start_pos, end_pos))
        lines_with_newlines.append(line)
        pos = end_pos + 1 # +1 for the newline character

    full_text = '\n'.join(lines_with_newlines)

    inputs = text_tokenizer(
        full_text,
        return_tensors='pt',
        return_offsets_mapping=True,
        truncation=True,
        max_length=6000,
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs['offset_mapping']

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits)
    predicted_labels = (probs > decision_boundary).nonzero(as_tuple=False)
    target_ids = predicted_labels[:, 1]

    if len(target_ids) == 0:
        continue

    attributions = explainer(input_ids, target_ids, device)
    predicted_probs = probs[0, target_ids].cpu().numpy()
    label_prob_pairs = [(tid.item(), predicted_probs[idx], attributions[:, idx])
                        for idx, tid in enumerate(target_ids)]
    label_prob_pairs_sorted = sorted(label_prob_pairs, key=lambda x: x[1], reverse=True)

    if label_prob_pairs_sorted:
        target_ids, predicted_probs, attributions = zip(*label_prob_pairs_sorted)
    else:
        target_ids, predicted_probs, attributions = ([], [], [])

    tokens = text_tokenizer.convert_ids_to_tokens(input_ids[0])
    offsets = offset_mapping[0].tolist()

    # For each predicted label, find and store evidence
    for idx, target_id in enumerate(target_ids):
        token_attributions = attributions[idx]
        prob = predicted_probs[idx]
        code = codes[target_id]

        token_info = []
        for (token, (start, end), attribution) in zip(tokens, offsets, token_attributions):
            if start == 0 and end == 0: # Skip special tokens
                continue
            token_text = full_text[start:end]
            token_info.append({
                'token': token_text,
                'start': start,
                'end': end,
                'attribution': attribution.item(),
            })

        if not token_info:
            continue
        
        # Accumulate attributions per line
        line_attributions = {}
        for t in token_info:
            if t['attribution'] < 0.0001: # Adjustable threshold for attribution (this value catches pretty much everything)
                continue
            # Find which line this token is in
            for line_idx, (line_start, line_end) in enumerate(line_spans):
                if t['start'] >= line_start and t['end'] <= line_end:
                    line_number = line_idx + 1
                    line_attributions[line_number] = line_attributions.get(line_number, 0.0) + t['attribution']
                    break

        if not line_attributions:
            continue
        
        # Sort lines by total attribution
        sorted_lines = sorted(line_attributions.items(), key=lambda x: x[1], reverse=True)
        evidence_lines = [ln for ln, _ in sorted_lines]
        evidence_attributions = [line_attributions[ln] for ln in evidence_lines]

        # Get spans and texts for each evidence line
        evidence_spans = []
        evidence_texts = []
        for line_number in evidence_lines:
            span_start = group_sorted.loc[group_sorted['line_number'] == line_number, 'start_index'].values[0]
            span_end = group_sorted.loc[group_sorted['line_number'] == line_number, 'end_index'].values[0]
            evidence_spans.append((span_start, span_end))
            line_text = group_sorted.loc[group_sorted['line_number'] == line_number, 'text'].values[0]
            line_text_no_tags = re.sub(r'<[^>]*>', '', str(line_text))
            evidence_texts.append(line_text_no_tags.strip())

        results.append({
            'note_id': note_id,
            'predicted_code': code,
            'predicted_code_probability': prob,
            'evidence_line_numbers': evidence_lines, # Ordered by attribution score (highest to lowest)
            'evidence_spans': evidence_spans,
            'evidence_texts': evidence_texts,
            'evidence_attributions': evidence_attributions,
        })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_filename}.csv", index=False)
results_df.to_parquet(f"{output_filename}.parquet", index=False)