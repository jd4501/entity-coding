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

def validate_input_file(filepath):
    """Validate the input entities CSV file exists and has correct format."""
    if not os.path.exists(filepath):
        print(f" Error: Input file not found: {filepath}")
        print(" Make sure you've run entity extraction first:")
        print("   python ner/extract_entities.py data/sample_data/sample_notes.parquet --output_file results/ner/sample_notes_entities.csv")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        required_cols = ['note_id', 'text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f" Error: Input file missing required columns: {missing_cols}")
            print(f"   Found columns: {list(df.columns)}")
            sys.exit(1)
        print(f" Input file validated: {len(df)} entities from {df['note_id'].nunique()} notes")
        return df
    except Exception as e:
        print(f" Error reading input file: {e}")
        sys.exit(1)

def ensure_output_directory(output_path):
    """Create output directory if it doesn't exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f" Created output directory: {output_dir}")

# TODO: Change to use Argparse
# Read post-NER file from command line
if len(sys.argv) > 1:
    filename = sys.argv[1]
    # Only strip .csv if it actually ends with .csv
    if filename.endswith('.csv'):
        filename = filename[:-4]
else:
    # Do not proceeed
    print("Usage: python infer_with_explanations.py <input_file.csv> [<output_file>]")
    print(" Please provide the input entities CSV file as the first argument.")
    sys.exit(1)

# Read output filename from command line (optional)
if len(sys.argv) > 2:
    output_filename = sys.argv[2]
    # Strip extensions properly
    if output_filename.endswith('.csv'):
        output_filename = output_filename[:-4]
    elif output_filename.endswith('.parquet'):
        output_filename = output_filename[:-8]
else:
    name = os.path.basename(filename)
    output_filename = f"../inferred_notes_with_evidence_{name}"

# Validate input and create output directory
input_filepath = f"{filename}.csv"
df = validate_input_file(input_filepath)
ensure_output_directory(f"{output_filename}.csv")


def validate_model_files():
    """Validate that all required model files exist."""
    model_path = Path('models/entityonly')
    tokenizer_path = Path('models/tokenizer_latest')
    
    required_files = [
        (model_path / 'target_tokenizer.json', 'ICD code mapping'),
        (model_path / 'config.yaml', 'Model configuration'),
        (model_path / 'best_model.pt', 'Trained model weights'),
        (tokenizer_path, 'Entity tokenizer')
    ]
    
    for file_path, description in required_files:
        if not file_path.exists():
            print(f" Error: {description} not found at {file_path}")
            print(" Make sure you've downloaded all models with: python data_download.py")
            sys.exit(1)
    
    print(" All required model files found")

# Validate model files exist
validate_model_files()

# Load entity model
print(" Loading ICD coding models...")
model_path = Path('models/entityonly')
target_codes_path = model_path / 'target_tokenizer.json'

# Load the index-to-code mapping used by the model
with open(target_codes_path, 'r') as file:
    codes = json.load(file)  

saved_config = OmegaConf.load(model_path / "config.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

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
# original 0.4040403962135315
decision_boundary = 0.4040403962135315 # This is the tuned boundary for the entity-only model, but feel free to raise/lower it

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
            if t['attribution'] < 0.0001: # Adjustable threshold for token attribution (this value catches pretty much everything)
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

# Print completion summary
print("\n ICD coding inference completed successfully!")
print(f" Processed {df['note_id'].nunique()} notes with {len(df)} entities")
print(f" Generated {len(results_df)} ICD code predictions")
print(f" Results saved to:")
print(f"   • {output_filename}.csv")
print(f"   • {output_filename}.parquet")

print(f"\n You can now examine the results in the CSV/Parquet files.")
print(f"   Each prediction includes evidence spans and attribution scores.")
