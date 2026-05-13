# Code evidence reproduction

What this page covers: how to reproduce the MDACE evidence-overlap numbers reported in Table 8 and the MDACE classification metrics in Table 12 and Appendix B, for both the entity-only model (the main contribution) and the full-text PLM-CA baseline. These recipes require MDACE plus the relevant credentialed MIMIC inputs. The two flows share the MDACE preparation step but use different inference scripts and different evidence-aggregation downstream of inference.

## Prerequisites

Stage the PLM-CA raw inputs under `external/plm_ca/data/raw/` as described in [What data do I need?](inference.md#what-data-do-i-need): MIMIC-III, MIMIC-IV, MIMIC-IV-Note, and the vendored MDACE annotation tree under `data/raw/MDace/`. The MDACE evidence evaluation uses the MIMIC-III note text, then maps evidence labels into the ICD-10 code space used by the MIMIC-IV-trained coding models.

Download the model checkpoints for the flow you are running:

```bash
python data_download.py --models ner,ac,entity-only,fulltext --cleanup
( cd external/plm_ca && make download_roberta )
```

The `make download_roberta` step fetches the base RoBERTa-PM encoder that the released PLM-CA checkpoints load through their saved config. It is separate from `data_download.py --models roberta`, which fetches the distill-align variant used for NER/AC retraining.

From `external/plm_ca/`, build the processed MDACE-ICD10 inpatient splits:

```bash
python -m explainable_medical_coding.data.prepare_mimiciii data/raw data/processed
python -m explainable_medical_coding.data.prepare_mimiciv data/raw data/processed
python -m explainable_medical_coding.data.prepare_mdace data/raw data/processed
python -m explainable_medical_coding.data.make_mdace_icd10_inpatient
```

These commands are the conda-native equivalent of the upstream Makefile path. The inherited `make mimiciii mimiciv mdace` shorthand is Poetry-based; prefer the commands above in the supported `entitycoding` environment.

## Code evidence evaluation (entity-only vs MDACE)

This is the entity-only side of Tables 8 and 12 plus Appendix B.

1. **Build MDACE-DC split files.** Open `code_evidence/code_evidence_eval.ipynb` and run **Stage 1 (Preprocessing)** from the repo root. Set `mdace_dir` in that cell to `external/plm_ca/data/processed/mdace_icd10_inpatient` if it is not already set that way. The cell filters to discharge-summary reports and writes `mdace_{val,train,test}_DConly.parquet` in the notebook working directory. The commands below assume those files are in the repo root.
2. **Run NER + AC on the MDACE-DC parquets.** From the repo root:
   ```bash
   python ner/extract_entities.py mdace_val_DConly.parquet \
       --output-file results/ner/mdace_val_entities.csv --max_workers 5
   python ner/extract_entities.py mdace_train_DConly.parquet \
       --output-file results/ner/mdace_train_entities.csv --max_workers 5
   python ner/extract_entities.py mdace_test_DConly.parquet \
       --output-file results/ner/mdace_test_entities.csv --max_workers 5
   ```
3. **Run entity-only ICD inference.** From `external/plm_ca/`, write the output names expected by the notebook's default entity-only cells:
   ```bash
   python infer_with_explanations.py \
       ../../results/ner/mdace_val_entities.csv \
       ../../inferred_notes_with_evidence_mdace_val
   python infer_with_explanations.py \
       ../../results/ner/mdace_train_entities.csv \
       ../../inferred_notes_with_evidence_mdace_train
   python infer_with_explanations.py \
       ../../results/ner/mdace_test_entities.csv \
       ../../inferred_notes_with_evidence_mdace_test
   ```
4. **Evaluate.** Return to the notebook and run **Stage 2 (Threshold Tuning)** onwards with `model_name = "entityonly"` and `PROBABILITY_THRESHOLD = 0.4040403962135315`. The notebook tunes the attribution threshold on validation, then evaluates on the combined MDACE-DC train+test set, matching the manuscript protocol after validation has been consumed for threshold selection. Outputs are written under an `entityonly/` folder in the notebook working directory.

## Full-text evidence reproduction

The flow above evaluates evidence emitted by the _entity-only_ model. The paper also reports full-text PLM-CA baseline numbers in Table 8, Table 12, and Appendix B; reproducing those uses two additional scripts in `external/plm_ca/`. The full-text decision boundary `0.4141414165496826` matches `PROBABILITY_THRESHOLD` in `code_evidence_eval.ipynb`.

The scoring split is important: `infer_with_explanations_fulltext.py` does not produce the final Table 8 span scores by itself. It emits per-word candidate evidence; `merge_contiguous_spans.py` applies the val-tuned attribution threshold and collapses surviving adjacent words into character spans; the notebook then scores those spans against MDACE.

1. **Build MDACE-DC CSVs.** Use the same Stage 1 notebook cell as the entity-only flow, but uncomment the three `.to_csv(...)` lines before running it. `infer_with_explanations_fulltext.py` reads CSV, while the notebook keeps the parquet files for ground truth.
2. **Extract per-word evidence at a permissive threshold.** From `external/plm_ca/`:
   ```bash
   python infer_with_explanations_fulltext.py \
       ../../mdace_val_DConly ../../fft_inferred_notes_with_evidence_mdace_val_DConly 1e-6
   python infer_with_explanations_fulltext.py \
       ../../mdace_train_DConly ../../fft_inferred_notes_with_evidence_mdace_train_DConly 1e-6
   python infer_with_explanations_fulltext.py \
       ../../mdace_test_DConly ../../fft_inferred_notes_with_evidence_mdace_test_DConly 1e-6
   ```
3. **Tune the attribution threshold on val.** In Stage 2 of the notebook, switch the validation prediction path to `fft_inferred_notes_with_evidence_mdace_val_DConly.parquet`, set `model_name = "fulltext"` and `PROBABILITY_THRESHOLD = 0.4141414165496826`, then sweep for the attribution value that maximises partial-match F2 on val. The manuscript run selected `0.0013877551020408164`.
4. **Merge per-word outputs into contiguous spans** for train and test, using the val-derived threshold:
   ```bash
   python merge_contiguous_spans.py \
       ../../fft_inferred_notes_with_evidence_mdace_train_DConly \
       ../../fft_inferred_notes_with_evidence_mdace_train_DConly_merged \
       0.0013877551020408164  # example; use your val-tuned value
   python merge_contiguous_spans.py \
       ../../fft_inferred_notes_with_evidence_mdace_test_DConly \
       ../../fft_inferred_notes_with_evidence_mdace_test_DConly_merged \
       0.0013877551020408164
   ```
5. **Run Get Basic Metrics** onwards in the notebook. Point `preds_train` and `predicted_df_test` at the merged train/test files from step 4, keep `test_gt = pd.concat([test_docs, train_docs], axis=0)`, and run the remaining cells to produce the final full-text evidence numbers. The notebook writes the corresponding CSVs under `fulltext/` in the notebook working directory.
