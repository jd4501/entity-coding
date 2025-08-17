import pandas as pd

def process_data(folder):
    # File paths for train, validation, and test sets
    train_path = f"data/processed/mimiciv_icd10/{folder}/train.parquet"
    val_path = f"data/processed/mimiciv_icd10/{folder}/val.parquet"
    test_path = f"data/processed/mimiciv_icd10/{folder}/test.parquet"
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    
    # Remove special tokens from the text
    def strip_specials(row):
        text = row['text']
        text = text.replace('<procedure>', '')
        text = text.replace('<health_context>', '')
        text = text.replace('<disorder>', '')
        text = text.replace('<abnormal_finding>', '')
        text = text.replace('<medication>', '')
        return text.strip()
    
    def get_length(row):
        return len(row['text_stripped'].split())
    
    all_df['text_stripped'] = all_df.apply(strip_specials, axis=1)
    all_df['length'] = all_df.apply(get_length, axis=1)
    
    Q1 = all_df['length'].quantile(0.25)
    median = all_df['length'].median()
    Q3 = all_df['length'].quantile(0.75)
    IQR = Q3 - Q1
    total_length = all_df['length'].sum()
    
    print(f"{folder.upper()} DATA:")
    print(f"Median doc length: {median} with IQR {IQR} ({Q1} - {Q3})")
    print(f"Total words: {total_length}\n")
    
    return all_df

entities_df = process_data("entities")
fulltext_df = process_data("fulltext")
