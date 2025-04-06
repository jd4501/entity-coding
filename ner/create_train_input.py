#!/usr/bin/env python3
import csv
from collections import defaultdict
import random
import pandas as pd
import argparse
import sys

random.seed(1)

# For ablations
TOKENS_TO_REMOVE = [
    "<disorder>",
    "<medication>",
    "<procedure>",
    "<health_context>",
    "<abnormal_finding>",
]

# For ablations
REPLACEMENT_MAPPING = {
    "<disorder>": "(Disorder)",
    "<medication>": "(Medication)",
    "<procedure>": "(Procedure)",
    "<health_context>": "(Health context)",
    "<abnormal_finding>": "(Abnormal finding)",
}

def process_text(text, remove_tokens_flag=False, replace_tokens_flag=False):
    """
    Process the text by optionally removing or replacing tokens.
    If remove_tokens_flag is True, all occurrences of tokens in TOKENS_TO_REMOVE are removed.
    If replace_tokens_flag is True, tokens in REPLACEMENT_MAPPING are replaced with their corresponding values.
    """
    if remove_tokens_flag:
        for token in TOKENS_TO_REMOVE:
            text = text.replace(token, "")
    if replace_tokens_flag:
        for token, replacement in REPLACEMENT_MAPPING.items():
            text = text.replace(token, replacement)
    return text

def combine_text_by_note_id(input_csv, shuffle=False, remove_tokens_flag=False, replace_tokens_flag=False):
    """
    Reads the CSV file and groups text rows by note_id.
    
    Before combining:
      - Optionally, shuffles the order of texts if shuffle==True.
      - Optionally, processes each text to remove or replace specific tokens.
    
    Returns:
        A dictionary mapping note_id to the combined text.
    """
    note_texts = defaultdict(list)
    
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            note_id = row['note_id']
            text = row['text']
            
            # Process the text if removal or replacement flags are set
            text = process_text(text, remove_tokens_flag, replace_tokens_flag)
            
            note_texts[note_id].append(text)
    
    # If shuffle is enabled, randomize the order of texts for each note_id before combining
    combined = {}
    for note_id, texts in note_texts.items():
        if shuffle:
            random.shuffle(texts)
        combined[note_id] = "\n".join(texts)
    
    return combined

def replace_text_in_parquet(parquet_file, combined_texts, output_parquet_file):
    """
    Reads the Parquet file into a DataFrame and replaces its 'text' column with the combined text from the CSV (if available)
    based on note_id. The updated DataFrame is saved to a new Parquet file.
    
    Args:
        parquet_file (str): Path to the input Parquet file.
        combined_texts (dict): Dictionary mapping note_id to updated text.
        output_parquet_file (str): Path for saving the updated Parquet file.
        
    Returns:
        The updated DataFrame.
    """
    df_parquet = pd.read_parquet(parquet_file)
    
    if "note_id" not in df_parquet.columns:
        raise ValueError("The Parquet file must contain a 'note_id' column.")
    
    df_csv = pd.DataFrame(list(combined_texts.items()), columns=['note_id', 'updated_text'])
    
    df_merged = df_parquet.merge(df_csv, on='note_id', how='left')
    
    df_merged['text'] = df_merged['updated_text'].fillna(df_merged['text'])
    
    df_merged.drop(columns=['updated_text'], inplace=True)
    
    df_merged.to_parquet(output_parquet_file)
    print(f"Updated Parquet file saved to: {output_parquet_file}")
    return df_merged

def remove_note_ids_from_parquet(df, note_ids_to_remove, output_parquet_file):
    """
    Removes rows from the DataFrame where the note_id is in note_ids_to_remove and saves the filtered DataFrame to a new Parquet file.
    
    Args:
        df (DataFrame): The DataFrame to filter.
        note_ids_to_remove (list): List of note_ids to remove.
        output_parquet_file (str): Path for saving the filtered Parquet file.
        
    Returns:
        The filtered DataFrame.
    """
    df_filtered = df[~df['note_id'].isin(note_ids_to_remove)]
    df_filtered.to_parquet(output_parquet_file)
    print(f"Filtered Parquet file saved to: {output_parquet_file}")
    return df_filtered

def main():
    parser = argparse.ArgumentParser(
        description="Merge document texts from extracted entities into a MIMIC dataframe by note_id."
    )
    parser.add_argument("--entities", required=True, help="Path to the input extracted entities file.")
    parser.add_argument("--mimic_file", required=True, help="Path to the input MIMIC data file.")
    parser.add_argument("--output", required=True, help="Path to the output Parquet file with updated text.")
    parser.add_argument(
        "--remove_ids",
        nargs="*",
        default=None,
        help="Optional list of note_ids to remove from the final output (separate multiple IDs by spaces)."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle (randomize) texts before combining them for each note_id."
    )
    parser.add_argument(
        "--remove_tokens",
        action="store_true",
        help="Remove specific tokens (<disorder>, <medication>, <procedure>, <health_context>, <abnormal_finding>) from texts."
    )
    parser.add_argument(
        "--replace_tokens",
        action="store_true",
        help="Replace specific tokens with alternative labels in texts."
    )
    
    args = parser.parse_args()
    
    # Check that both removal and replacement tokens flags are not set simultaneously
    if args.remove_tokens and args.replace_tokens:
        print("Error: --remove_tokens and --replace_tokens cannot be used together.")
        sys.exit(1)
    
    combined_texts = combine_text_by_note_id(
        args.entities,
        shuffle=args.shuffle,
        remove_tokens_flag=args.remove_tokens,
        replace_tokens_flag=args.replace_tokens
    )
    print(f"Extracted and combined text for {len(combined_texts)} note_ids from CSV.")
    
    df_updated = replace_text_in_parquet(args.mimic_file, combined_texts, args.output)
    
    # (Optional): Remove specified note_ids from the updated DataFrame.
    if args.remove_ids:
        output_filtered = args.output.replace(".parquet", "_filtered.parquet")
        df_final = remove_note_ids_from_parquet(df_updated, args.remove_ids, output_filtered)
        print(f"Final dataset shape after removal: {df_final.shape}")
    else:
        print("No note_ids were removed from the updated Parquet file.")

if __name__ == "__main__":
    main()