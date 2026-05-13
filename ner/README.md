# Named Entity Recognition (NER)

The `ner/` directory hosts the clinical entity recogniser used in the project's
ICD-10 coding pipeline, plus the supporting scripts that prepare its training
data and reproduce the released annotation statistics. The recogniser detects
six entity categories (`normal_finding`, `abnormal_finding`, `disorder`,
`procedure`, `health_context`, `medication`), matching the label set in the
MIMIC-IV-Ext-EntityCoding release. Downstream stages then filter the detections
by assertion status (see `ac/`) and drop `normal_finding` (no ICD-10-CM/PCS
codes are assigned for normal findings), leaving the five-category entity-only
document fed to PLM-CA (see `external/plm_ca/`) with type-tag tokens
`<disorder>`, `<medication>`, `<procedure>`, `<health_context>`, and
`<abnormal_finding>`.

The contents split into three loosely independent flows:

1. **Inference flow.** Clean a clinical note and extract assertion-filtered
   entities for the entity-only coding model.
2. **Training-data preparation and model training.** Sample MIMIC-IV-Note
   discharge summaries for annotation, convert released annotations into the
   training-notebook format, and train the deployed NER checkpoint.
3. **Release statistics and inter-annotator agreement.** Recompute the Table 1
   summary statistics from the released CSVs and reproduce the paper's IAA
   numbers.

End-to-end reproduction commands and the broader pipeline narrative live in
[`docs/inference.md`](../docs/inference.md) and [`docs/reproduce.md`](../docs/reproduce.md);
the root [`README.md`](../README.md) is the landing page. This file is intended as an
index plus per-script notes for the accessory entry points.

## Files

### Inference flow

| Path                    | Purpose                                                                                                                                                                                                             | Default inputs                                                                                                                               | Default outputs                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `extract_entities.py`   | Clean each note, run the RoBERTa-PM token classifier, classify assertion status per detection, and write the assertion-filtered entity rows used downstream.                                                        | Notes CSV or parquet with `note_id` and `text` columns; `data/models/ner_model/`; `data/models/ac_model/`                                    | `results/ner/output_entities.csv` (columns `note_id,start_index,end_index,text`)                    |
| `clean_documents.py`    | Stand-alone document cleaner. Applies the same preprocessing as `extract_entities.py` without running NER. Useful for inspecting cleaned text or for piping notes through a different model.                        | Same notes CSV or parquet shape                                                                                                              | CSV with `note_id,text`                                                                             |
| `document_cleaning.py`  | Cleaning helpers (line-wrap repair, whitespace normalisation, PyRuSH sentence splitting with the medspaCy section detector) shared by `extract_entities.py` and `clean_documents.py`. Importable; not a CLI script. | n/a                                                                                                                                          | n/a                                                                                                 |
| `create_train_input.py` | Group entity rows by note, join them with newlines, and write a parquet shard whose `text` column is the entity-only document. Drives the entity-only training inputs for `external/plm_ca/train_plm_entities.py`.  | Entity CSV from `extract_entities.py`; full-text MIMIC-IV ICD-10 parquet (e.g. `external/plm_ca/data/processed/mimiciv_icd10/train.parquet`) | Entity-only parquet (e.g. `external/plm_ca/data/processed/mimiciv_icd10/entity-only/train.parquet`) |
| `rush_rules.tsv`        | PyRuSH sentence-splitter rules used by `document_cleaning.py`. Adapted from medspaCy/PyRuSH resources and locally tuned for MIMIC-IV notes. Reference asset; do not edit ad hoc.                                   | n/a                                                                                                                                          | n/a                                                                                                 |
| `section_patterns.json` | medspaCy section header patterns used by `document_cleaning.py`. Adapted from medspaCy resources and locally tuned for MIMIC-IV notes. Reference asset.                                                             | n/a                                                                                                                                          | n/a                                                                                                 |

### Training-data preparation and model training

| Path                           | Purpose                                                                                                                                                                                                                                                                                                                                                        | Default inputs                                                                                                              | Default outputs                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `ner_dataset_creation.py`      | One-off script used during paper preparation to choose the MIMIC-IV discharge summaries that were then sent for manual NER annotation. Samples by `curr_service` to match the source-data service distribution and excludes notes already covered by the SNOMED CT Entity Linking Challenge. Kept for provenance; not part of the release-time inference flow. | `data/mimic-iv-note/services.csv`, `data/mimic-iv-note/discharge.csv`, `data/mimic-iv-note/mimic-iv_notes_training_set.csv` | `data/ner/ner_dataset_notes.csv`, `data/ner/ner_dataset_notes.parquet` |
| `prepare_ner_training_data.py` | Convert the MIMIC-IV-Ext-EntityCoding release CSVs into the Label Studio JSON shape consumed by `ner_model_training.ipynb`.                                                                                                                                                                                                                                    | `data/mimic-iv-ext-entitycoding/entity_annotations.csv`, `data/mimic-iv-ext-entitycoding/mimic-iv_notes_subset.csv`         | `results/ner/ner_ac_label_studio_format.json`                          |
| `ner_model_training.ipynb`     | Train the deployed NER checkpoint and reproduce Table 3 (5-fold CV). Reads the Label Studio JSON above and exports the model to `data/models/ner_model/`.                                                                                                                                                                                                      | `results/ner/ner_ac_label_studio_format.json`                                                                               | `data/models/ner_model/`                                               |

### Release statistics and inter-annotator agreement

| Path                         | Purpose                                                                                                                                                                              | Default inputs                                                                                                                                                                  | Default outputs |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `compute_release_stats.py`   | Recompute the Table 1 summary statistics (entity counts by type, entities per document, words per entity, assertion counts) from the released CSVs.                                  | `data/mimic-iv-ext-entitycoding/entity_annotations.csv`, `data/mimic-iv-ext-entitycoding/assertion_annotations.csv`, `data/mimic-iv-ext-entitycoding/mimic-iv_notes_subset.csv` | Stdout          |
| `inter_annotator_agreement/` | Token-level Kappa and entity-level F1 between the two annotators on the IAA subset. See `inter_annotator_agreement/README.md` for inputs, expected output, and the subset breakdown. | `inter_annotator_agreement/annotations/texts*.txt` and `author{1,2}*.json` (gated)                                                                                              | Stdout          |

## Inputs and Where They Come From

Most inputs to these scripts are gated clinical-data artefacts and live in
local-only stage areas under `data/`. Each path below is described in
`data/README.md`; this section is a quick map of what each script expects.

- **`data/mimic-iv-ext-entitycoding/`** holds the MIMIC-IV-Ext-EntityCoding
  PhysioNet release: note text, entity spans, assertion sentences, and the
  annotation guidelines. Consumed by `prepare_ner_training_data.py`,
  `compute_release_stats.py`, and (via `ac/`) the assertion classifier.
- **`data/mimic-iv-note/`** is an ad hoc staging directory for raw MIMIC-IV-Note
  source files (`discharge.csv`, `services.csv`) and the SNOMED CT Entity
  Linking Challenge note set. Consumed by `ner_dataset_creation.py`.
- **`data/ner/`** is the local cache of NER training inputs and the entity
  CSVs derived from MIMIC-IV-Note. It can be regenerated by re-running
  `extract_entities.py` and `prepare_ner_training_data.py`.
- **`data/models/ner_model/`** and **`data/models/ac_model/`** are the
  deployed NER and assertion classifier checkpoints, populated by
  `data_download.py` from the Google Drive bundle.
- **`data/sample_data/sample_notes.csv`** is the small synthetic note set used
  by the smoke-test command in the top-level README.

The MIMIC-IV-Ext-EntityCoding release files, MIMIC-IV-Note staging files, IAA
annotation files, and trained model checkpoints are all subject to PhysioNet
credentialed-access terms or to the project's own credentialed distribution
channels. None of them are committed to this repository.

## Script Notes

### `extract_entities.py`

The first stage of the inference pipeline. For each note it:

1. Cleans the text with `document_cleaning.py` (line-wrap repair, whitespace
   normalisation, PyRuSH sentence splitting, section-aware sentence joining).
2. Runs the RoBERTa-PM token-classification NER model from
   `data/models/ner_model/` on batches of cleaned sentences (joined with
   newlines and capped at roughly 500 tokens per inference call). Sentences
   that exceed the cap on their own are recursively split on punctuation.
3. Wraps each detected entity in `<entity> ... <entity>` and runs the
   assertion classifier from `data/models/ac_model/` to label each
   detection. Detections labelled `absent` or `hypothetical` are discarded;
   `possible` is retained for disorders, abnormal findings, and health
   contexts; `not associated with patient` is retained for disorders and
   abnormal findings (used as a family-history signal); procedures and
   medications keep only `present` detections.
4. Drops every `normal_finding` (no ICD-10-CM/PCS code is assigned to
   normal observations) and wraps each retained entity span in its type
   tag (`<disorder>`, `<medication>`, `<procedure>`, `<health_context>`,
   `<abnormal_finding>`).

The output CSV has columns `note_id,start_index,end_index,text`, where `text`
is the type-tagged entity surface form and the offsets index the cleaned note.

Useful flags:

- `--output_file PATH`: output CSV path (default `results/ner/output_entities.csv`).
- `--max_workers N`: parallel worker processes (default 5). Each worker loads
  its own NER and AC model, so scale down on smaller GPUs.
- `--save-formatted-texts`: also write the cleaned note text under
  `results/formatted_texts/`.
- `--save-ner-docs`: also write per-note displaCy HTML under
  `results/ner/docs_with_ner/`.
- `--force`: ignore the resume-from-existing-output check and reprocess every
  note from scratch. By default the script skips notes already present in the
  output CSV, which makes long batches restartable.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`.

### `clean_documents.py`

Same cleaning pipeline as `extract_entities.py` but without NER or assertion
classification. The output CSV quotes all fields so embedded newlines in
cleaned notes round-trip through `pandas.read_csv`. Useful flags:

- `--output_file PATH` (required): output CSV path.
- `--filter_ids PATH`: optional CSV listing `note_id` values to keep.
- `--max_workers N`: parallel worker processes (default 4).
- `--force`: discard any existing output instead of resuming from it.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`.

### `document_cleaning.py`

Library module. Two entry points:

- `clean_document_text(text, nlp)`: returns the cleaned note as a single
  string.
- `clean_document_sentences(text, nlp)`: returns the cleaned note plus the
  per-sentence list and character offsets used to map NER spans back into the
  cleaned document.

Both accept an optional pre-built spaCy pipeline from `build_cleaning_nlp()`.
If omitted, a fresh pipeline is built per call; pass one in and reuse it
across notes when processing in batch.

The local `rush_rules.tsv` and `section_patterns.json` files are intentional.
They are adapted from the rule resources bundled with medspaCy and PyRuSH, but
this project keeps local copies because the rules were manually edited for
MIMIC-IV discharge notes. The edits improve sentence segmentation around
clinical-note line breaks and improve detection of MIMIC-IV section headers.
Treat these files as MIMIC-IV-tuned project assets rather than as generic
medspaCy defaults. Upstream attribution remains with medspaCy/PyRuSH and the
original PyRuSH rule authors; `rush_rules.tsv` retains its upstream University
of Utah Apache License, Version 2.0 header.

### `create_train_input.py`

Sits between `extract_entities.py` and `external/plm_ca/train_plm_entities.py`.
It groups entity rows by `note_id`, joins each note's entities with newlines,
and replaces the `text` column of the upstream MIMIC-IV ICD-10 parquet with
the entity-only document. The rest of the parquet schema (codes, splits, ids)
is preserved, so the PLM-CA dataset loader treats the entity-only document
exactly like a full-text note.

Required flags:

- `--entities PATH`: extracted entities CSV from `extract_entities.py`.
- `--mimic_file PATH`: full-text MIMIC-IV ICD-10 parquet shard with `note_id`
  and `text` columns.
- `--output PATH`: entity-only output parquet path.

Optional flags reproduce the manuscript's entity-token and ordering ablation
splits (manuscript Section 4.4):

- `--remove_tokens`: drop the five entity type tokens from each entity row.
- `--replace_tokens`: replace the five entity type tokens with plain-text
  labels (e.g. `<disorder>` becomes `(Disorder)`). Mutually exclusive with
  `--remove_tokens`.
- `--shuffle`: shuffle entities within each note before joining them.
- `--remove_ids ID [ID ...]`: drop specific notes from the output.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`.

For entity-type subset experiments, pre-filter the extracted-entities CSV to
the desired entity categories before passing it here.

### `ner_dataset_creation.py`

Recreates the historical service-balanced sample of MIMIC-IV discharge
summaries that were sent for manual NER annotation. The defaults (target size
389, random state 1) match the paper-era settings. Sampling is per individual
`curr_service` value; a hard-coded list of surgical service codes is used only
to log the surgical vs non-surgical proportion as a sanity check. Excludes any
note that already appears in the SNOMED CT Entity Linking Challenge training
set so the new entity dataset covers different MIMIC notes. The target of 389
is calibrated against the post-exclusion ceiling rounding, and yields the
manuscript's 400 annotated notes on the paper data.

### `prepare_ner_training_data.py`

Joins each note's processed text with its entity spans and writes the Label
Studio JSON shape used by `ner_model_training.ipynb`. Notes follow the order
in the input notes CSV; entity spans within each note are sorted by `start`,
`end`, and `entity_label` so reruns are easy to compare. Labels are passed
through unchanged.

Each output record looks like:

```json
{
  "id": "<note_id>",
  "text": "<note text>",
  "label": [{ "start": 0, "end": 10, "labels": ["disorder"] }]
}
```

### `ner_model_training.ipynb`

Reads `results/ner/ner_ac_label_studio_format.json`, runs 5-fold cross
validation, prints the Table 3 metrics, and exports the deployed checkpoint to
`data/models/ner_model/`. Run cells top-to-bottom with the `entitycoding`
environment active. The notebook uses RoBERTa-base-PM-M3-Voc-distill-align-hf
as the encoder; populate it from the Google Drive bundle via
`data_download.py --models roberta` if it is not already present.

### `compute_release_stats.py`

Reads the released entity, assertion, and notes CSVs and prints the Table 1
summary: entity counts by type, entities per document (median and IQR), words
per entity, and assertion counts by class. All inputs are overridable via CLI
flags; the defaults assume the release files are unpacked under
`data/mimic-iv-ext-entitycoding/`.

### `inter_annotator_agreement/`

Reproduces the F1 = 0.77 / Cohen's Kappa = 0.81 numbers reported in paper
section 3.1. The package has its own README (`inter_annotator_agreement/README.md`)
with input layout, the exact command, expected output, and notes on the
scoring methodology.

## Dependencies

The top-level `environment.yml` defines the supported `entitycoding` conda
environment. Non-stdlib imports used by the NER scripts are `pandas`, `numpy`,
`torch`, `transformers`, `spacy`, `medspacy`, `scikit-learn`, `tqdm`, and
`pyarrow`. The IAA scoring script only needs `scikit-learn`.
