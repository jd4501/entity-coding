# Less is More: Explainable and Efficient ICD Code Prediction with Clinical Entities

## Paper: https://aclanthology.org/2025.acl-long.1489/

This repository contains the code and resources for the paper **"Less is More: Explainable and Efficient ICD Code Prediction with Clinical Entities"**. The work incorporates Named Entity Recognition (NER) and Assertion Classification (AC) to detect medical mentions releavnt for clinical coding, using them in downstream code prediction and evidence extraction.

---

## Overview

This repository builds upon and modifies the [PLM-CA repository](https://github.com/JoakimEdin/explainable-medical-coding), with paper paper: [An Unsupervised Approach to Achieve Supervised-Level Explainability in Healthcare Records](https://aclanthology.org/2024.emnlp-main.280/). It adds:

- **NER & AC Models**: Identifies clinical entities (e.g., disorders, procedures) and their assertion status (e.g., present, absent/negated)
- **Entity-based ICD Code Prediction**: Instead of feeding the entire note text into the model, we consolidate the detected entities into a structured input.
- **Evidence Extraction**: Provides evidence spans (entities) that support each predicted ICD code, improving transparency and interpretability of the coding process.

Note: It uses this [PLM-CA commit](https://github.com/JoakimEdin/explainable-medical-coding/commit/8269cc7246b88fa5dd299191713ed7475b908537). 

---

## Installation

**1. Clone this repository**

**2. Create Conda Environment**  
  Navigate to this repository’s root folder and run:

  ```
  # Using Mamba (recommended for faster installation):
  mamba env create -f environment.yml
  
  # Or using Conda:
  conda env create -f environment.yml
  
  # Activate the environment:
  conda activate entitycoding
  ```

**Note:** The environment.yml includes dependencies from the PLM-CA module, and the new NER/AC components. You can safely ignore the installation step from PLM-CA (`make setup`).

**3. Obtain PLM-CA Resources**  
  - Navigate to the PLM-CA module (`cd external/plm-ca`)
  - Ensure to activate the conda environment (`conda activate entitycoding`)
  - Follow their setup instructions to download MIMIC-IV, MIMIC-IV-Note and MIMIC-III (see `external/plm_ca/README.md`, Section `Setup`) to `external/plm_ca/data/raw`. 
  - You do _not_ need to run `make_prepare_everything` (since the PLM-CA trained coding models are not needed for this work). and you do _not_ need to make another environment.
  - Instead, run the Makefiles up to (and including) `make download_roberta` to obtain the MIMIC data and RoBERTa encoder (you can choose all or one of `make mimiciii`, `make mimiciv`, `make mdace`). 

**4. Download NER/AC/Coding Models** 

This repository requires several pre-trained models that are hosted externally due to GitHub file size limits. Navigate back to the root directory, then use the automated download script to fetch all required models:

**Activate the conda environment first:**
```bash
conda activate entitycoding
```

**Download with cleanup (removes archive files after extraction):**
```bash
python data_download.py --cleanup
```

Note: The download paths are stored in `config/download_config.yaml`. Commenting out a given path will skip downloading it. 

### What Gets Downloaded

The script automatically downloads and extracts the following models:

- **NER Model** (433MB) → `data/models/ner_model/`
  - Used for Named Entity Recognition in clinical notes
- **AC Model** (434MB) → `data/models/ac_model/`  
  - Used for Assertion Classification (present/absent/uncertain etc.)
- **RoBERTa-base-PM-M3-Voc-distill-align** (~470MB) → `data/models/RoBERTa-base-PM-M3-Voc-distill-align-hf/`
  - Pre-trained biomedical language model
- **Entity-only ICD-10 Coding Model** (~1.3GB) → `external/plm_ca/models/entityonly/`
  - ICD code prediction model trained on entity-based input
- **Full-text ICD-10 Coding Model** (~1.3GB) → `external/plm_ca/models/fulltext/`
  - ICD code prediction model trained on full text notes
- **Tokenizer with Entity Classes** → `external/plm_ca/models/tokenizer_latest/`
  - Custom tokenizer that handles the NER entity tokens (the full-text model does not need this)

**Note:** The full-text ICD-10 coding model is currently commented out in the download config but can be enabled if desired.

## Usage

#### 1. Prepare Input Documents

You will need a Parquet or CSV file with two columns:

- **note_id**: A unique identifier for each clinical note
- **text**: The raw text of the clinical note

An example file is provided for testing: `data/sample_data/sample_notes.csv` (GPT-4o generated clinical notes). The MIMIC note downloads from the PLM-CA module already contain these columns and are compatible. However, they are downloaded in their full text form. See Section `Generate Entity Documents` for details on how to create the entity-only documents from the MIMIC-III/IV dataset. 

### End-to-end NER, AC and ICD coding

```bash
# Activate environment 
conda activate entitycoding

# Run complete pipeline with default settings
python run_pipeline.py data/sample_data/sample_notes.csv

# Or customize the output file path/name, number of parallel workers for NER/AC and toggle HTML visualization of codes and evidence
python run_pipeline.py data/sample_data/sample_notes.csv [output_path] --max_workers 4 --visualize
```

**What the pipeline script does:**
1. **Entity Extraction**: Identifies and filters clinical entities (disorders, procedures, etc.) using NER and AC models. 
2. **ICD Coding**: Predicts ICD-10 codes based on extracted entities with evidence attribution
3. **Output Generation**: Creates both CSV and Parquet files with ICD code results and evidence. 
4. **Visualization (OPTIONAL)**: Saves HTML outputs showing a note with predicted codes and evidence spans. Requires the --visualize flag. 

### Manual Steps

If you prefer to run each step individually:

#### 1. Entity Extraction

**Activate the conda environment first:**
```bash
conda activate entitycoding
```

**Run entity extraction:**
```bash
python ner/extract_entities.py data/sample_data/sample_notes.csv --output_file results/ner/sample_notes_entities.csv --max_workers 4
```

**Important Notes:**
- Adjust `--max_workers` based on your GPU memory, for parallel processing of notes. 5-6 works with ~12GB VRAM, as a guide. Each worker process loads its own copy of the NER and AC models.
- Add `--save-formatted-texts` if you plan to visualise the codes+evidence later (saves the formatted documents to which NER/AC is applied, with pre-processing applied)
- Add `--save_ner_docs` if you'd like to save the docs with NER/AC spans highlighted (HTML format), via `displacy`. 
- The script is designed to detect note_id's that have already been processed. This is helpful if running on large document sets, but means you'll need to remove already processed files if you'd like to re-run with one of the flags above applied. 

#### 2. ICD Coding Inference

With the extracted entities (.csv file from the previous step), you can run ICD coding inference with evidence extraction:

**Navigate to PLM-CA module and run inference:**
```bash
cd external/plm_ca
python infer_with_explanations.py ../../results/ner/sample_notes_entities.csv ../../results/coded/sample_notes_coded
```
Argument 1 is the input file (only .csv accepted). Argument 2 is the output file (including .csv or .parquet at the end is optional - both will be saved)

Note: This script currently only accepts .csv files as input. It does, however, save outputs to both .csv and .parquet (regardless of whether you provide .csv or .parquet in the output extension). 

**Output files:**
- `results/coded/sample_notes_coded.csv` - Main results with ICD codes, probabilities, and evidence
- `results/coded/sample_notes_coded.parquet` - Same data in Parquet format

## PLM-CA Training

If you wish to train or fine-tune models within the PLM-CA framework using entity-based notes:

### Generate Entity-only Documents

In the PLM-CA repository, after running `make mimiciv`, your processed train/val/test data will be in `explainable_medical_coding/data/processed/mimiciv_icd10`. You'll need to run NER/AC on each of the train/test/val files with `extract_entities.py`.

Next, consolidate all entities for each note into a single document and replace the original full text document. Running `create_train_input.py` on each pair of files (post-NER and AC + the original MIMIC file) will create the entity-only documents for training. Example:

`python ner/create_train_input.py --entities results/ner/mimic-iv-train.csv --mimic_file external/plm-ca/data/processed/mimiciv_icd10/train.parquet --output results/train.parquet`

### (Optional) Ablation Testing

`--remove_tokens` removes special tokens indicating entity types.
`--replace_tokens` replaces special tokens with plain text.
`--shuffle` shuffles entities and headings in the note.

### Run Training

Once the train/val/test files have been processed, navigate to the PLM-CA repo `cd external/plm-ca` and run training: `poetry run python train_plm_entities.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10`.

IMPORTANT: The HuggingFace integration can lead to previous versions of the dataset being cached for further use. If this cached data is not removed, you may inadvertenly train the model on the wrong version of the data. This cache path varies by OS, but should resemble `.cache/huggingface/datasets/mdace_inpatient_icd10`. These folders can be safely deleted. 

## Code Evidence Evaluation

To evaluate code evidence (using [MDACE](https://github.com/3mcloud/MDACE)):

### Prepare MDACE Data

If you ran `make mdace` in the PLM-CA repo, the processed MDACE files will be located in `external/plm-ca/data/processed/mdace_icd10_inpatient`.

### NER & AC on MDACE Notes

Run the same `extract_entities.py` steps on the MDACE train/val/test notes to obtain entity-level outputs.

### Evidence Extraction

Run `infer_with_explanations.py` to get code predictions and evidence for each note. The arguments are the input file, and output filename.

### Threshold Tuning & Evaluation

In the `code_evidence_eval.ipynb` notebook (found in `code_evidence/`), set the file paths for the predicted outcomes from the previous script. Then run the cells to generate an evidence evaluation report that includes metrics on how accurately the model’s entity-based evidence aligns with reference data.
