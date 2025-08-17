import pandas as pd
import numpy as np

# File paths
services_file = 'services.csv' # Download from the hosp directory in the base MIMIC-IV dataset
notes_file = 'discharge.csv' # Download from MIMIC-IV note
snomed_ner_data_file = 'mimic-iv_notes_training_set.csv' # Download from SNOMED CT Entity Linking Challenge (Physionet)

# Output paths
new_dataset_filename = 'data/ner/ner_dataset_notes'

# Step 1: Load data and re-attach service type information

services_df = pd.read_csv(services_file) 
all_notes_df = pd.read_csv(notes_file) 

# Load the SNOMED-annotated subset of notes and extract note IDs
snomed_notes_df = pd.read_csv(snomed_ner_data_file)
snomed_note_ids = snomed_notes_df['note_id'].tolist()

# Filter for notes that are included in the SNOMED set
snomed_included_notes = all_notes_df[all_notes_df['note_id'].isin(snomed_note_ids)]

# Merge to attach service type information based on subject and admission IDs
snomed_notes_with_service = pd.merge(
    snomed_included_notes,
    services_df,
    on=('subject_id', 'hadm_id'),
    how='left'
).drop_duplicates('note_id')


# Step 2: Compute proportions of service types in SNOMED-included and general discharge summaries

# Calculate counts and proportions for SNOMED-included notes
included_counts = snomed_notes_with_service.groupby('curr_service')['note_id'].count()
included_proportions = included_counts / included_counts.sum()

included_stats = pd.DataFrame({
    'count_included': included_counts,
    'proportion_included': included_proportions
})

# For general discharge summaries, filter to discharge summaries and merge service info
discharge_notes = all_notes_df[all_notes_df['note_type'] == 'DS']
discharge_notes_with_service = pd.merge(
    discharge_notes,
    services_df,
    on=('subject_id', 'hadm_id'),
    how='left'
).drop_duplicates('note_id')

general_counts = discharge_notes_with_service.groupby('curr_service')['note_id'].count()
general_proportions = general_counts / general_counts.sum()

general_stats = pd.DataFrame({
    'count_general': general_counts,
    'proportion_general': general_proportions
})

comparison_stats = pd.merge(
    included_stats,
    general_stats,
    left_index=True,
    right_index=True,
    how='outer'
).fillna(0)

print("Comparison of service type counts and proportions:")
print(comparison_stats)


# Optional Step 3 - Evaluate how well the SNOMED dataset adheres wrt surgical cases

# service types considered surgical
surgical_services = ['CSURG', 'NSURG', 'ORTHO', 'PSURG', 'SURG', 'TRAUM', 'TSURG', 'VSURG', 'ENT']

prop_surgical_included = comparison_stats.loc[surgical_services, 'proportion_included'].sum()
prop_surgical_general = comparison_stats.loc[surgical_services, 'proportion_general'].sum()

print(f"Proportion of surgical services (SNOMED included): {prop_surgical_included}")
print(f"Proportion of surgical services (General discharge notes): {prop_surgical_general}")

# Note: was ~ 26.5% (snomed) vs 27.8% (general mimic) - but some subcategories overlooked 


# Step 4: Create a new set of notes to match the true (general) proportions

# Set the target total number of additional samples desired (note: due to rounding this number inflates to 400)
target_total_samples = 389

# Calculate the target count per service type based on the general proportions
comparison_stats['target_count'] = comparison_stats['proportion_general'] * target_total_samples

# Determine the additional number of samples needed per service type
comparison_stats['additional_samples_needed'] = comparison_stats['target_count']
comparison_stats['additional_samples_needed'] = comparison_stats['additional_samples_needed'].clip(lower=0)

# Round up to ensure minority classes are sampled
comparison_stats['additional_samples_needed'] = np.ceil(comparison_stats['additional_samples_needed']).astype(int)

# Display the calculated sample needs for each service type
print("Calculated additional samples needed per service type:")
print(comparison_stats[['count_included', 'proportion_included', 'count_general', 'proportion_general', 'target_count', 'additional_samples_needed']])

total_additional_needed = comparison_stats['additional_samples_needed'].sum()
print(f"Total additional samples needed: {total_additional_needed}")


# Sample extra notes (from general discharge notes not already in the SNOMED set)

# Note: A decision was made here to increase NER breadth by not overlapping with notes in the SNOMED challenge
sampled_extra_notes_list = []

for service, samples_needed in comparison_stats['additional_samples_needed'].items():
    if samples_needed > 0:
        service_notes = discharge_notes_with_service[
            (discharge_notes_with_service['curr_service'] == service) &
            (~discharge_notes_with_service['note_id'].isin(snomed_note_ids))
        ]
        sampled_rows = service_notes.sample(
            n=int(samples_needed),
            replace=False if samples_needed <= len(service_notes) else True,
            random_state=1
        )
        sampled_extra_notes_list.append(sampled_rows)

sampled_extra_notes = pd.concat(sampled_extra_notes_list, ignore_index=True)
sampled_extra_notes = sampled_extra_notes[['note_id', 'subject_id', 'hadm_id', 'curr_service', 'text']]

# Save the extra notes for annotation
sampled_extra_notes.to_csv(f"{new_dataset_filename}.csv", index=False)
sampled_extra_notes.to_parquet(f"{new_dataset_filename}.parquet", index=False)


# Compare dataset proportions

combined_notes_df = pd.concat([snomed_notes_with_service, sampled_extra_notes], ignore_index=True)
final_counts = combined_notes_df.groupby('curr_service')['note_id'].count()
final_proportions = final_counts / final_counts.sum()

# Create a comparison table between original, general, and final proportions
proportion_comparison = pd.DataFrame({
    'snomed_proportion': included_proportions,
    'general_proportion': general_proportions,
    'new_dataset_proportion': final_proportions
}).fillna(0)

print("Final comparison of service type proportions:")
print(proportion_comparison)
