# Less is More: Explainable and Efficient ICD Code Prediction with Clinical Entities

---

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Usage](#usage)
   - [NER and AC Inference](#ner-and-ac-inference)
   - [ICD Coding Inference](#icd-coding-inference)
   - [PLM-CA Training](#plm-ca-training)
   - [Code Evidence Evaluation](#code-evidence-evaluation)
5. [Project Structure](#project-structure)

---

## Introduction

This contains the code and resources for the paper **"Less is More: Explainable and Efficient ICD Code Prediction with Clinical Entities"**. The work incorporates Named Entity Recognition (NER) and Assertion Classification (AC) to detect medical mentions releavnt for clinical coding, using them in downstream code prediction and evidence extraction.

---

## Overview

This repository builds upon and modifies the [PLM-CA repository](https://github.com/JoakimEdin/explainable-medical-coding). It adds:

- **NER & AC Models**: Identifies clinical entities (e.g., disorders, procedures) and determines whether the entity is present, absent, or otherwise relevant in the clinical note.
- **Entity-based ICD Code Prediction**: Instead of feeding the entire note text into the model, we consolidate the detected entities into a structured input.
- **Evidence Extraction**: Provides evidence spans (entities) that support each predicted ICD code, improving transparency and interpretability of the coding process.

---

## Installation

1. **Clone PLM-CA Repository**  
   First, clone the [PLM-CA repository](https://github.com/JoakimEdin/explainable-medical-coding) and follow their setup instructions.

   - You do _not_ need to run `make_prepare_everything` (since the PLM-CA trained coding models are not needed for this work).
   - Instead, run the Makefiles up to (and including) `make download_roberta`.

2. **Clone This Repository**  
   Clone or download **this** repository.

3. **Create Conda Environment**  
   Navigate to this repository’s root folder and run:
   ```
   conda env create -f environment.yml
   conda activate entitycoding
   pip install -r requirements.txt
   ```

## Download and Unzip Models

Download the trained models (NER, AC, and RoBERTa-PM) and unzip them as follows:

- **NER Model** [Link](https://drive.google.com/file/d/1GZwp5E0yK-q-17JWznZx4eLR9zKVsJK6/view?usp=sharing) → `data/models/ner_model`
- **AC Model** [Link](https://drive.google.com/file/d/1WEqsBbTSibrGmq_O0zEsU5rTmohVaMgQ/view?usp=sharing) → `data/models/ac_model`
- **RoBERTa-base-PM-M3-Voc-distill-align** [Link](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz) → `data/models/RoBERTa-base-PM-M3-Voc-distill-align-hf`
- **Entity-only ICD-10 Coding Model** [Link](https://drive.google.com/file/d/1nFVnQ29nx8R-p4RSMdQ0kAP4Mod3vZKj/view?usp=sharing) → `changes/models/entityonly`
- **Tokenizer with Entity Classes** [Link](https://drive.google.com/file/d/1STXM3w0PIpWOuN0d7X2NhCWTHuNPYBW2/view?usp=sharing) → `changes/models/tokenizer_latest`
- **Full-text ICD-10 Coding Model** [Link](https://drive.google.com/file/d/17FFuK7VsyxdaqIMdiOqeh3WvE4JXr6b0/view?usp=sharing) → `changes/models/fulltext`

## Integrate Changes

Copy the contents of the `changes` folder from this repository into the previously cloned `explainable-medical-coding` folder, overwriting files where necessary. This step applies the modifications required for the entity-based approach.

## Usage

### NER and AC Inference

#### Prepare Input

You will need a Parquet file with two columns:

- **note_id**: A unique identifier for each clinical note
- **text**: The raw text of the clinical note

An _Example file_ for demonstration been provided (created via GPT-4o): `ner/sample_notes.parquet`

#### Run Extraction

Use `extract_entities.py` to detect named entities and classify their assertions:

`python ner/extract_entities.py ner/sample_notes.parquet --output_file ner/sample_notes_entities.csv --max_workers 5`

Adjust `--max_workers` based on your GPU memory (5 was tested on 12GB).

## ICD Coding Inference

With the extracted entities (.csv file from the previous step), you can run ICD coding inference with evidence extraction:

### Navigate to PLM-CA Repo

`cd explainable-medical-coding`

### Run Inference

`python infer_with_explanations.py ../ner/sample_notes_entities.csv --output_file ../entity_notes`

## PLM-CA Training

If you wish to train or fine-tune models within the PLM-CA framework using entity-based notes:

### Generate Entity Documents

Consolidate all entities for each note into a single document (replacing the original full text). In the PLM-CA repository, after running `make mimiciv`, your processed train/val/test data will be in `explainable_medical_coding/data/processed/mimiciv_icd10`. Running `create_train_input.py` on each pair of of files (post-NER and AC + the original MIMIC file) will create the entity-only documents for training.

`python ner/create_train_input.py --entities ner/sample_notes_entities.csv --mimic_file ner/sample_notes.parquet --output entity_notes.parquet`

### (Optional) Ablation Testing

`--remove_tokens` removes special tokens indicating entity types.
`--replace_tokens` replaces special tokens with plain text.
`--shuffle` shuffles entities and headings in the note.

### Run Training

Once the train/val/test files have been processed, navigate to the PLM-CA repo `cd explainable-medical-coding` and run training: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10`.

## Code Evidence Evaluation

To evaluate code evidence (using [MDACE](https://github.com/3mcloud/MDACE)):

### Prepare MDACE Data

If you ran make `mdace_icd9` in the PLM-CA repo, the processed MDACE files will be located in `explainable_medical_coding/data/processed/mdace_icd10_inpatient`.

### NER & AC on MDACE Notes

Run the same `extract_entities.py` steps on the MDACE train/val/test notes to obtain entity-level outputs.

### Evidence Extraction

Use `infer_with_explanations.py` to get code predictions and evidence for each note.

### Threshold Tuning & Evaluation

In the `code_evidence_eval.ipynb` notebook (found in `code_evidence/`), set the file paths for the predicted outcomes. Then run the cells to generate an evidence evaluation report that includes metrics on how accurately the model’s entity-based evidence aligns with reference data.

## Project Structure

Below is a simplified outline of the directories and key files:

entity_coding/
├── data/
│ └── models/ # Downloaded/trained NER, AC, and coding models
├── ner/
│ ├── extract_entities.py # Script to run NER & AC inference
│ ├── create_train_input.py # Script to prepare entity-based training/validation/test data for PLM-CA
│ ├── ner_model_training.ipynb # Notebook for training the NER model
│ └── ... # Additional files for preprocessing
├── ac/
│ └── ... # Files for merging data and training the AC model
├── changes/
│ └── ... # Modified files to overwrite in explainable_medical_coding
├── explainable_medical_coding/
│ └── ... # Cloned from https://github.com/JoakimEdin/explainable-medical-coding
├── code_evidence/
│ └── code_evidence_eval.ipynb # Jupyter notebook for evidence evaluation
├── results/
│ └── ... # Results for code classification and evidence outputs
├── environment.yml # Conda environment definition
├── requirements.txt # Additional pip dependencies
└── README.md # Project documentation (this file)
