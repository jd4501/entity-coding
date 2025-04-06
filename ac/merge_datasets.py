import pandas as pd
from html import unescape
import re
import string
from sklearn.model_selection import train_test_split

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and filter datasets
base_train = pd.read_csv('ac/data/2010_train.csv')
base_train = base_train[base_train['assertion'] != 'conditional']
base_test = pd.read_csv('ac/data/2010_test.csv')
base_test = base_test[base_test['assertion'] != 'conditional']

mimic = pd.read_csv('ac/data/mimic_assertion_data.csv')
mimic = mimic[mimic['assertion'] != 'present']

i2b2_2012 = pd.read_csv('ac/data/i2b2_2012_merged.csv')
i2b2_2012 = i2b2_2012[i2b2_2012['assertion'] != 'present']
i2b2_2012['text'] = i2b2_2012['text'].apply(unescape)

# Normalize text
base_train['text_norm'] = base_train['text'].apply(normalize_text)
base_test['text_norm'] = base_test['text'].apply(normalize_text)
i2b2_2012['text_norm'] = i2b2_2012['text'].apply(normalize_text)

# Remove overlaps
overlap_2012_with_train = pd.merge(base_train, i2b2_2012, on='text_norm', how='inner')
i2b2_2012 = i2b2_2012[~i2b2_2012['text_norm'].isin(overlap_2012_with_train['text_norm'])]

overlap_2012_with_test = pd.merge(base_test, i2b2_2012, on='text_norm', how='inner')
i2b2_2012 = i2b2_2012[~i2b2_2012['text_norm'].isin(overlap_2012_with_test['text_norm'])]

# Combine training and test, mark sources
base_train['source'] = 'train'
base_test['source'] = 'test'
combined_df = pd.concat([base_train, base_test], ignore_index=True).drop_duplicates(subset='text')

target_col = 'assertion'
total_size = len(combined_df)
test_size = int(total_size * 0.2)
train_size = int(total_size * 0.7)
val_size = total_size - train_size - test_size

# Split off 20% test from original test
test_data_from_test, remaining_test = train_test_split(
    combined_df[combined_df['source'] == 'test'],
    test_size=(len(base_test) - test_size),
    stratify=combined_df[combined_df['source'] == 'test'][target_col],
    random_state=1
)

train_filtered = combined_df[combined_df['source'] == 'train'][['text', target_col]]
remaining_data = pd.concat([train_filtered, remaining_test], ignore_index=True)

train_data, val_data = train_test_split(
    remaining_data,
    test_size=val_size / (train_size + val_size),
    stratify=remaining_data[target_col],
    random_state=1
)

print(f"Total size (i2b2 2010 only - rebalanced): {total_size}")
print(f"Train size (i2b2 2010 only - rebalanced): {len(train_data)}")
print(f"Validation size (i2b2 2010 only - rebalanced): {len(val_data)}")
print(f"Test size (i2b2 2010 only - rebalanced): {len(test_data_from_test)}")

train_data = train_data[['text', 'assertion']]
val_data = val_data[['text', 'assertion']]
test_data_from_test = test_data_from_test[['text', 'assertion']]

# Save intermediate train
train_data.to_csv('ac/data/train_i2b2_only.csv', index=False)

# Prepare i2b2 2012
i2b2_2012 = i2b2_2012[['text', 'assertion']]

# Combine mimic and i2b2 2012
expanded_training = pd.concat([mimic, i2b2_2012], ignore_index=True)

# Normalize text
combined_df['text_norm'] = combined_df['text'].apply(normalize_text)
expanded_training['text'] = expanded_training['text'].apply(unescape)
expanded_training['text_norm'] = expanded_training['text'].apply(normalize_text)
expanded_training = expanded_training.drop_duplicates(subset='text_norm')

# Remove overlaps
common_texts = set(expanded_training['text_norm']).intersection(set(combined_df['text_norm']))
expanded_training = expanded_training[~expanded_training['text_norm'].isin(common_texts)]

# Add new annotations
new_data = pd.read_csv('ac/data/all_new_assertions.csv')
expanded_training = expanded_training[['text', 'assertion']]
expanded_training = pd.concat([expanded_training, new_data], ignore_index=True)
expanded_dataset = expanded_training.dropna()

# Split expanded dataset
expanded_train, expanded_testval = train_test_split(
    expanded_dataset, test_size=0.3,
    stratify=expanded_dataset[target_col],
    random_state=1
)

expanded_test, expanded_val = train_test_split(
    expanded_testval, test_size=0.65,
    stratify=expanded_testval[target_col],
    random_state=1
)

# Final merges
final_train = pd.concat([train_data, expanded_train], ignore_index=True)
final_val = pd.concat([val_data, expanded_val], ignore_index=True)
final_test = pd.concat([test_data_from_test, expanded_test], ignore_index=True)

final_train = final_train[['text', 'assertion']]
final_val = final_val[['text', 'assertion']]
final_test = final_test[['text', 'assertion']]

train_val = pd.concat([final_train, final_val], ignore_index=True)
train_val.to_csv('ac/data/train_val.csv', index=False)

all_data = pd.concat([final_train, final_val, final_test], ignore_index=True)
all_data.to_csv('ac/data/all_assertion_data.csv', index=False)

# Check final splits
final_train.to_csv('ac/data/train_expanded.csv', index=False)
final_val.to_csv('ac/data/val_expanded.csv', index=False)
final_test.to_csv('ac/data/test_expanded.csv', index=False)

total_size_expanded = len(final_train) + len(final_test) + len(final_val)
print(f"Total expanded size: {total_size_expanded}")
print(f"Train size: {len(final_train)}")
print(f"Validation size: {len(final_val)}")
print(f"Test size: {len(final_test)}")
