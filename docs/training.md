# Training entity-only PLM-CA from scratch

What this page covers: the recipe to retrain the entity-only ICD coding model from MIMIC-IV credentialed inputs, including the dataset-path contract the PLM-CA loader expects, the entity-extraction pass, and the Hydra command that launches training. The release ships a trained checkpoint via `python data_download.py --models entity-only`; this page is for users who want to reproduce or modify the coding-model training procedure itself. For the AC model's training pipeline see [`ac/README.md`](../ac/README.md).

## Overview

The release ships a trained entity-only checkpoint, but the coding-model retraining path can be rerun with credentialed MIMIC-IV inputs and the downloaded NER/AC checkpoints. The contract: PLM-CA's loader (`external/plm_ca/explainable_medical_coding/datasets/mimiciv_icd10.py`) reads `external/plm_ca/data/processed/mimiciv_icd10/{train,val,test}.parquet`. To train on entity-only documents you produce entity-only parquets and place them at exactly those paths.

Minimum model setup for this page:

```bash
python data_download.py --models ner,ac --cleanup
( cd external/plm_ca && make download_roberta )
```

The first command supplies the entity extractor and assertion classifier used in Stage 2. The second fetches the base RoBERTa-PM encoder that PLM-CA training loads from `external/plm_ca/models/roberta-base-pm-m3-voc-hf/`.
`python data_download.py --models roberta` is not a substitute for this second command; that selector downloads the distill-align variant used by NER/AC retraining.

**Stage 1: regenerate full-text MIMIC-IV-ICD10 splits.** This requires the credentialed MIMIC-IV inputs already staged (see [What data do I need?](inference.md#what-data-do-i-need)). From `external/plm_ca/`, run the two data-prep modules that the upstream `make mimiciv` target wraps:

```bash
python -m explainable_medical_coding.data.prepare_mimiciv data/raw data/processed
python -m explainable_medical_coding.data.make_mimiciv_icd10
```

These materialise `data/processed/mimiciv_icd10/{train,val,test}.parquet`. The inherited `make mimiciv` target is broader and Poetry-based; the two commands above are the supported conda-native path for this repo and avoid building the ICD-9 MIMIC-IV shard that entity-only ICD-10 retraining does not need.

**Stage 2: extract entities for each split.** From the repo root:

```bash
python ner/extract_entities.py external/plm_ca/data/processed/mimiciv_icd10/train.parquet \
    --output-file results/ner/mimiciv-train.csv --max_workers 5
# Repeat for val.parquet and test.parquet.
```

**Stage 3: build entity-only training documents.** Group the entities for each note, replace the `text` column of the upstream parquet with the joined entity document, and write a new parquet:

```bash
python ner/create_train_input.py \
    --entities results/ner/mimiciv-train.csv \
    --mimic_file external/plm_ca/data/processed/mimiciv_icd10/train.parquet \
    --output external/plm_ca/data/processed/mimiciv_icd10/entity-only/train.parquet
# Repeat for val and test.
```

Notes producing zero entity rows are dropped rather than passed through as full text (emitting raw text would silently defeat the entity-only pipeline). On MIMIC-IV-ICD10 this drops exactly one train note (`14187825-DS-9`, a duplicate-report placeholder); val and test are unaffected. The released train parquet has 89,097 rows. Useful flags on `create_train_input.py`: `--remove_tokens`, `--replace_tokens`, `--shuffle` (Table 10 ablations), and `--remove_ids ID1 ID2 ...` (drop specific notes from the output without rerunning NER).

**Stage 4: swap parquets and clear the dataset cache.** PLM-CA reads from the top-level paths, so place your entity-only files there. Keep the full-text shards under `fulltext/` and the entity-only shards under `entity-only/`; that side-by-side layout is also what the Table 5 length script expects.

The example below uses POSIX shell commands. On Windows PowerShell, use the equivalent `New-Item -ItemType Directory`, `Move-Item`, `Copy-Item`, and `Remove-Item` commands.

```bash
cd external/plm_ca/data/processed/mimiciv_icd10
mkdir -p fulltext entity-only
mv train.parquet val.parquet test.parquet fulltext/
cp entity-only/train.parquet train.parquet
cp entity-only/val.parquet val.parquet
cp entity-only/test.parquet test.parquet
rm -rf ~/.cache/huggingface/datasets/mimiciv_icd10  # essential after a swap
```

**Stage 5: launch training.** From `external/plm_ca/`:

```bash
python train_plm_entities.py experiment=mdace_icd9_code/plm_icd \
    gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10
```

The trailing `data=mimiciv_icd10` override is required; without it Hydra silently falls back to the experiment file's ICD-9 default.

**Retraining clobbers `external/plm_ca/models/tokenizer_latest/`.** `train_plm_entities.py` re-saves the tokenizer (with the five entity special tokens added) to that path on launch, overwriting the released tokenizer that `data_download.py --models entity-only` puts there for inference. If you need the released variant later, back it up before retraining (`cp -r external/plm_ca/models/tokenizer_latest external/plm_ca/models/tokenizer_released`) and restore it when you're done.

With the default callbacks, output checkpoints land under `external/plm_ca/models/<wandb-run-id>/`. To redirect them, copy `external/plm_ca/.env.example` to `external/plm_ca/.env` and edit `EXPERIMENT_PATH` (its default value `models` matches the in-code fallback, so the copy is only needed for overrides).

For wandb-related stumbles during launch, see [troubleshooting](troubleshooting.md).
