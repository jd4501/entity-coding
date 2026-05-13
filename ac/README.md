# Assertion Classification (AC)

The AC pipeline reproduces Table 4 of the paper. It fine-tunes a RoBERTa-PM
classifier on four assertion sources: i2b2 2010, i2b2 2012, MIMIC-III
bvanaken labels, and the in-house MIMIC-IV-Ext-EntityCoding annotations. At
inference time `ner/extract_entities.py` wraps each candidate entity in
`<entity> ... <entity>` and uses the trained classifier to predict whether the
concept is present, absent, possible, hypothetical, or associated with someone
else.

None of the AC training datasets are committed to this repository. `ac/sources/`
and `ac/data/` are user-populated staging directories. Download each source
dataset listed under "Source Data" below into `ac/sources/`, and the
`prepare_*` and `merge_datasets.py` scripts will regenerate intermediate and
split CSVs into `ac/data/`. Both directories may contain gated clinical text
or challenge data and must not be committed.

## Files

| Path                            | Purpose                                                                 | Default inputs                                                                                                                                      | Default outputs                                                                   |
| ------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ac/common.py`                  | Shared path anchors, logging setup, CSV checks, and safe tar extraction | n/a                                                                                                                                                 | n/a                                                                               |
| `ac/prepare_i2b2_2010.py`       | Parse i2b2 2010 assertion tarballs                                      | `ac/sources/i2b2_2010/*.tar.gz`                                                                                                                     | `ac/data/2010_train.csv`, `ac/data/2010_test.csv`                                 |
| `ac/prepare_i2b2_2012.py`       | Parse i2b2 2012 Temporal Relations XML tarballs                         | `ac/sources/i2b2_2012/*.tar.gz`                                                                                                                     | `ac/data/2012_train.csv`, `ac/data/2012_test.csv`, `ac/data/i2b2_2012_merged.csv` |
| `ac/prepare_mimic_iii.py`       | Join bvanaken assertion labels to MIMIC-III note text                   | `ac/sources/mimic_iii/`                                                                                                                             | `ac/data/mimic_assertion_data.csv`                                                |
| `ac/prepare_our_annotations.py` | Convert released MIMIC-IV-Ext-EntityCoding assertion sentences          | `data/mimic-iv-ext-entitycoding/assertion_sentences.csv`                                                                                            | `ac/data/our_new_assertions.csv`                                                  |
| `ac/merge_datasets.py`          | Filter, dedupe, and split all AC sources                                | the five prepared CSVs listed above                                                                                                                 | `ac/data/{train,val,test}_expanded.csv`                                           |
| `ac/train_ac_model.py`          | Train and export the deployment AC model                                | split CSVs plus `data/models/RoBERTa-base-PM-M3-Voc-distill-align-hf/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf/` | `data/models/ac_model/`                                                           |
| `ac/train_eval_model_cv.py`     | Run 5-fold cross-validation for Table 4 metrics                         | same split CSVs and base model as `train_ac_model.py`                                                                                               | logs and fold checkpoints under `ac/data/_cv_fold_checkpoints/`                   |

## Source Data

You must download every source dataset listed below before running the AC
pipeline; none of these files ship with the repository, and each download
requires its own license or credentialing step (DBMI Data Portal for the i2b2
challenges, PhysioNet credentialing for MIMIC-III and MIMIC-IV-Ext-EntityCoding,
public GitHub for the bvanaken label CSVs). The scripts default to the paths
shown here, but each input can be overridden with CLI flags such as
`--train-tar`, `--noteevents`, or `--sentences`.

### i2b2 2010 Relations Challenge

Download from the DBMI Data Portal after registration and DUA approval:

- Landing page: <https://portal.dbmi.hms.harvard.edu/projects/n2c2-2010/>
- Project index: <https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/>

Place these files under `ac/sources/i2b2_2010/`:

```text
ac/sources/i2b2_2010/concept_assertion_relation_training_data.tar.gz
ac/sources/i2b2_2010/test_data.tar.gz
ac/sources/i2b2_2010/reference_standard_for_test_data.tar.gz
```

### i2b2 2012 Temporal Relations Challenge

Download from the DBMI Data Portal:

- Landing page: <https://portal.dbmi.hms.harvard.edu/projects/n2c2-2012/>

Place these two tarballs under `ac/sources/i2b2_2012/`:

```text
ac/sources/i2b2_2012/2012-07-15.original-annotation.release.tar.gz
ac/sources/i2b2_2012/2012-08-23.test-data.groundtruth.tar.gz
```

`prepare_i2b2_2012.py` ports the mapping logic from
`data/i2b2_2012/get_samples2.py` in
<https://github.com/bionlplab/assertion_classification_jbi2022>. You do not
need to clone that repository to run the AC pipeline; clone it only if you
want to inspect the upstream parser for verification.

### MIMIC-III bvanaken Assertions

`prepare_mimic_iii.py` combines two sources:

- Public labels from <https://github.com/bvanaken/clinical-assertion-data/tree/main/labels>
- MIMIC-III Clinical Database v1.4 `NOTEEVENTS.csv`, which requires PhysioNet credentialed access

Download the four label CSVs:

```bash
cd ac/sources/mimic_iii/bvanaken_labels
for f in discharge_summaries_labels.csv nursing_labels.csv \
         physician_labels.csv radiology_labels.csv; do
    curl -sSfL -o "$f" \
        "https://raw.githubusercontent.com/bvanaken/clinical-assertion-data/main/labels/$f"
done
```

Download `NOTEEVENTS.csv.gz` from
<https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz>, decompress it,
and place the CSV here:

```text
ac/sources/mimic_iii/NOTEEVENTS.csv
ac/sources/mimic_iii/bvanaken_labels/discharge_summaries_labels.csv
ac/sources/mimic_iii/bvanaken_labels/nursing_labels.csv
ac/sources/mimic_iii/bvanaken_labels/physician_labels.csv
ac/sources/mimic_iii/bvanaken_labels/radiology_labels.csv
```

This decompressed MIMIC-III CSV is specific to the AC pipeline. PLM-CA data
preparation uses its own raw-data layout under `external/plm_ca/data/raw/`.

### MIMIC-IV-Ext-EntityCoding

The AC pipeline reads only one file from the PhysioNet release:

```text
data/mimic-iv-ext-entitycoding/assertion_sentences.csv
```

If you are running other parts of this project as well, download the full
release into `data/mimic-iv-ext-entitycoding/`. If you only want to train the
AC model, you can fetch `assertion_sentences.csv` on its own and skip the rest
of the bundle. See `data/README.md` for the release schema and license. This
dataset inherits PhysioNet credentialing requirements from MIMIC-IV-Note.

## End-to-End Commands

Run commands from the repository root with the `entitycoding` conda environment
active.

Training steps also require the RoBERTa-PM initialization downloaded by
`python data_download.py --models roberta`.

Steps 1-4 each consume a different source dataset (i2b2 2010, i2b2 2012,
MIMIC-III plus bvanaken labels, and the MIMIC-IV-Ext-EntityCoding PhysioNet
release respectively). Obtain and stage all four under `ac/sources/` and
`data/mimic-iv-ext-entitycoding/` before running this section. "Source Data"
above lists portals, DUAs, and the exact target paths per source.

```bash
mkdir -p ac/data

# Step 1: parse i2b2 2010 tarballs.
python ac/prepare_i2b2_2010.py --cleanup

# Step 2: parse i2b2 2012 tarballs.
python ac/prepare_i2b2_2012.py --cleanup

# Step 3: build MIMIC-III assertion rows from bvanaken labels and NOTEEVENTS.
python ac/prepare_mimic_iii.py

# Step 4: build assertion rows from MIMIC-IV-Ext-EntityCoding.
python ac/prepare_our_annotations.py

# Step 5: merge, dedupe, and split all four sources.
python ac/merge_datasets.py

# Step 6a: train and export the deployment AC model.
python ac/train_ac_model.py --no-wandb

# Step 6b: optional 5-fold CV for Table 4.
python ac/train_eval_model_cv.py --no-wandb
```

Expected generated files:

```text
ac/data/2010_train.csv
ac/data/2010_test.csv
ac/data/2012_train.csv
ac/data/2012_test.csv
ac/data/i2b2_2012_merged.csv
ac/data/mimic_assertion_data.csv
ac/data/our_new_assertions.csv
ac/data/train_expanded.csv
ac/data/val_expanded.csv
ac/data/test_expanded.csv
data/models/ac_model/
```

`--cleanup` removes the extracted i2b2 working directories after parsing.
Omit it when you need to inspect the unpacked challenge files. All scripts
also accept `--log-level {DEBUG,INFO,WARNING,ERROR}`.

## Script Notes

`prepare_i2b2_2010.py` extracts the 2010 tarballs, walks the train
`{beth,partners}/ast/*.ast` files plus the test `.ast` files, finds each
concept in the corresponding `.txt` file, and writes `(text, assertion)` rows.
The source labels are kept verbatim. `merge_datasets.py` drops `conditional`.

`prepare_i2b2_2012.py` parses XML `<EVENT .../>` elements and maps modality
plus polarity to this project label space: `FACTUAL+POS` to `present`,
`FACTUAL+NEG` to `absent`, `POSSIBLE` to `possible`, and `CONDITIONAL` or
`HYPOTHETICAL` to `hypothetical`. This mirrors the upstream jbi2022 parser.

`prepare_mimic_iii.py` streams the large MIMIC-III `NOTEEVENTS.csv`, retains
only rows referenced by the bvanaken label files, wraps each span in
`<entity>` markers, and crops to the enclosing sentence. The label-file order
is fixed as `nursing`, `discharge_summaries`, `physician`, `radiology` because
that order reproduces the paper-era split membership.

`prepare_our_annotations.py` reads
`data/mimic-iv-ext-entitycoding/assertion_sentences.csv` and writes
`our_new_assertions.csv`. It maps `not_associated_with_patient` to
`associated_with_someone_else`, matching the i2b2 2010 label vocabulary used
by the deployed AC model.

`merge_datasets.py` drops 2010 `conditional` rows, drops `present` rows from
MIMIC-III and i2b2 2012, removes normalized-text duplicates, appends the
MIMIC-IV-Ext-EntityCoding rows, and creates the final train, validation, and
test splits. The default path preserves paper split behavior. `--deterministic`
sorts before each split for future order-invariant reruns, but it is not the
paper-exact split path.

`train_ac_model.py` trains the single deployment model with defaults matching
the paper setup: 12 epochs, batch size 8, learning rate `3e-5`, max length
512, patience 4, and seed 1. It saves the model, tokenizer, and label encoder
in the format consumed by `ner/extract_entities.py`.

`train_eval_model_cv.py` reports 5-fold cross-validation metrics over the
pooled split CSVs. It does not export a deployment model.

## Reproduction Fidelity

Once every source dataset listed above has been downloaded into the documented
locations, the preparation pipeline regenerates:

- `2010_train.csv`, `2010_test.csv`, `i2b2_2012_merged.csv`,
  `mimic_assertion_data.csv`, and `our_new_assertions.csv` with the expected
  row counts and schemas.
- `train_expanded.csv`, `val_expanded.csv`, and `test_expanded.csv` with row
  counts 22,889 / 4,534 / 4,531.

The MIMIC-III reconstruction includes one row with empty text. The upstream
bvanaken label for that row has `end_index < start_index`, so the cropping
window is empty; the row is preserved unchanged to match the paper-era input.
