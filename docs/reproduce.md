# Reproducing the manuscript

What this page covers: how manuscript artefacts in Douglas et al. (2025) map onto this repository where possible, plus the per-table recipes (commands, prerequisites, expected outputs). Some metric tables can be recomputed from downloaded released-checkpoint predictions; other artefacts require credentialed clinical data, retraining, paper-time logs, or manual review. Those limits are called out below.

Data access shorthand: Table 7 can be recomputed from the released checkpoint prediction feathers that `data_download.py` fetches. The committed per-code supplementary CSVs can be inspected without credentialed data; regenerating them needs those feathers plus the local training parquet used for `train_frequency`. The annotation, training, length, and MDACE evidence recipes require the relevant credentialed clinical inputs to be staged locally; MDACE annotations are public, but the note text they annotate comes from MIMIC-III.

## Paper artefact map

| Paper artefact                                                                                                                                                                                                    | Where it lives                                                               | How to reproduce                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Table 1 (annotation statistics)                                                                                                                                                                                   | PhysioNet annotation release staged under `data/mimic-iv-ext-entitycoding/`  | `python ner/compute_release_stats.py` after the credentialed release files are local                                                                                                                                                 |
| Annotation IAA (kappa 0.81, exact-span F1 0.77)                                                                                                                                                                   | `ner/inter_annotator_agreement/`                                             | `python ner/inter_annotator_agreement/compute_iaa.py ner/inter_annotator_agreement/annotations/texts.txt ner/inter_annotator_agreement/annotations/author1.json ner/inter_annotator_agreement/annotations/author2.json -d '#######'` after the IAA inputs are local |
| Table 2 (NER + AC hyperparameters)                                                                                                                                                                                | Constants in `ac/train_ac_model.py` and the NER notebook                     | Reference values, fixed in source                                                                                                                                                                                                    |
| Table 3 (NER 5-fold CV)                                                                                                                                                                                           | `ner/ner_model_training.ipynb`                                               | Requires the credentialed annotation release. First run `python ner/prepare_ner_training_data.py`, then run the notebook top to bottom                                                                                               |
| Table 4 (AC 5-fold CV)                                                                                                                                                                                            | `ac/train_eval_model_cv.py`                                                  | Requires the AC source corpora. See [`ac/README.md`](../ac/README.md) for the full preparation flow                                                                                                                                  |
| Table 5 (document-length reduction)                                                                                                                                                                               | `external/plm_ca/get_length_of_docs.py`                                      | See [Table 5 recipe](#table-5-recipe) below                                                                                                                                                                                          |
| Table 6 (training time per epoch)                                                                                                                                                                                 | Wandb logs from `external/plm_ca/train_plm_{entities,fulltext}.py`           | Logs are not committed; see [Hardware and disk](inference.md#hardware-and-disk) for context                                                                                                                                          |
| Table 7 (ICD coding metrics)                                                                                                                                                                                      | `external/plm_ca/eval_metrics.py`, `external/plm_ca/permutation_testing.py`  | See [Table 7 recompute](#table-7-recompute-and-significance-test) below                                                                                                                                                              |
| Table 8 (MDACE evidence overlap)                                                                                                                                                                                  | `code_evidence/code_evidence_eval.ipynb` (Stage 2 "Get Basic Metrics"); full-text also uses `infer_with_explanations_fulltext.py` + `merge_contiguous_spans.py` | See [Code evidence evaluation](evidence.md#code-evidence-evaluation-entity-only-vs-mdace), [Full-text evidence reproduction](evidence.md#full-text-evidence-reproduction), and [Code evidence scoring quick guide](#code-evidence-scoring-quick-guide) |
| Table 9 (qualitative superset examples)                                                                                                                                                                           | Manuscript-only manual analysis                                              | Not scripted                                                                                                                                                                                                                         |
| Table 10 (pipeline ablations)                                                                                                                                                                                     | Flags on `ner/create_train_input.py` for the token/order rows; the "no AC" row has no shipped flag | See [Tables 10 and 11 ablation recipes](#tables-10-and-11-ablation-recipes) below                                                                                                                                                    |
| Table 11 (entity-type subsets)                                                                                                                                                                                    | Manual pre-filtering of extracted entity rows before `create_train_input.py`  | See [Tables 10 and 11 ablation recipes](#tables-10-and-11-ablation-recipes) below                                                                                                                                                    |
| Table 12 (MDACE classification metrics)                                                                                                                                                                           | `code_evidence/code_evidence_eval.ipynb` (Stage 4)                           | See [Code evidence evaluation](evidence.md#code-evidence-evaluation-entity-only-vs-mdace) and [Full-text evidence reproduction](evidence.md#full-text-evidence-reproduction)                                                         |
| Appendix A (annotation guidelines)                                                                                                                                                                                | `data/mimic-iv-ext-entitycoding/annotation_guidelines.md`                    | Ships with the PhysioNet release once published                                                                                                                                                                                      |
| Appendix B (MDACE classification)                                                                                                                                                                                 | Same notebook as Table 12                                                    | See [Code evidence evaluation](evidence.md#code-evidence-evaluation-entity-only-vs-mdace) and [Full-text evidence reproduction](evidence.md#full-text-evidence-reproduction)                                                         |
| Per-code metrics CSVs (`results/per_code_metrics_{entityonly,fulltext}.csv`)                                                                                                                                      | `external/plm_ca/per_code_metrics.py`                                        | See [Per-code metrics recompute](#per-code-metrics-recompute) below                                                                                                                                                                  |
| Figure 1 (pipeline schematic)                                                                                                                                                                                     | [`img/workflow_diagram.png`](../img/workflow_diagram.png)                    | Shown under the README intro; no generation script is shipped                                                                                                                                                                        |
| Section 5.5 manual error analyses (200-sample false-positive relevance categorisation; qualitative subdivision of TP "No Overlap" cases into synonym/abbreviation, equally relevant, less relevant, insufficient) | Manual review of MDACE output samples                                        | Manual review; not scripted (see note below)                                                                                                                                                                                         |

**A note on the Section 5.5 manual analyses.** Both pieces (the 200-sample false-positive relevance categorisation and the qualitative subdivision of TP "No Overlap" cases) are manual reviews; no labels are committed. The underlying No Overlap pairs are derivable from `results/{entityonly,fulltext}_code_evidence_compared.csv` by re-running cell 9 of `code_evidence/code_evidence_eval.ipynb`, but the qualitative labels themselves are not stored.

## Table 5 recipe

Compares median word count of the full-text MIMIC-IV discharge summaries against the entity-only documents produced by the pipeline. Requires credentialed MIMIC-IV access; see [What data do I need?](inference.md#what-data-do-i-need).

1. Follow [Training entity-only PLM-CA from scratch](training.md) Stage 1 to materialise `external/plm_ca/data/processed/mimiciv_icd10/{train,val,test}.parquet` (full-text).
2. Run `python ner/extract_entities.py` over each split, then `python ner/create_train_input.py` to produce `external/plm_ca/data/processed/mimiciv_icd10/entity-only/{train,val,test}.parquet`.
3. Keep a copy of the original full-text shards under `external/plm_ca/data/processed/mimiciv_icd10/fulltext/{train,val,test}.parquet` as shown in [Training](training.md).
4. From `external/plm_ca/`, `python get_length_of_docs.py` prints median, IQR, and total word count for both trees (entity tags are stripped before whitespace tokenisation). It expects `data/processed/mimiciv_icd10/{entity-only,fulltext}/{train,val,test}.parquet`.

## Table 7 recompute and significance test

`python data_download.py --models entity-only,fulltext` populates `external/plm_ca/models/{entityonly,fulltext}/predictions_{validation,test}.feather`, which the recompute and significance scripts read by default.

Defaults inside `permutation_testing.py` are wired to those released feathers so reviewers can recompute the Table 7 significance test directly. The shipped script defaults to 1,000 permutation rounds for paper-equivalent numbers; pass `--num-permutations 5` for a fast smoke test. The paper-time output (1,000 permutations, about 6 hours in the original run environment) is committed at `results/paper_permutation_test_results.txt` for comparison.

From `external/plm_ca/`:

```bash
python permutation_testing.py
# or, for a quick wiring check:
python permutation_testing.py --num-permutations 5
```

For a fast metric-only check (no permutations), from `external/plm_ca/`:

```bash
python eval_metrics.py --model entityonly --split test
python eval_metrics.py --model fulltext   --split test
```

This prints all eight Table 7 metrics: `f1_micro`, `f1_macro`, `precision@8`, `precision@15`, `exact_match_ratio`, `map`, `auc_micro`, `auc_macro`. Macro AUC iterates `roc_curve` per code, so a full test-set run takes a few minutes. `--predictions <path>` and `--threshold <float>` accept arbitrary feathers.

The original PLM-ICD row in Table 7 is a literature-reference row from Edin et al. (2023), averaged over their reported runs. This repository recomputes the two PLM-CA rows produced by this project: full-text and entity-only.

## Per-code metrics recompute

If you want to inspect ICD-code-level performance directly (for example to look at how training frequency relates to per-code F1, or which codes the entity-only model handles better than the full-text baseline), open the committed supplementary tables `results/per_code_metrics_{entityonly,fulltext}.csv`. Each row gives per-ICD-code precision, recall, F1, TP, FP, FN, `test_count`, and `train_frequency` plus a `long_title` gloss. These exist precisely so reviewers can drill into code-level behaviour the manuscript itself does not break out. To regenerate them from `external/plm_ca/`:

```bash
python per_code_metrics.py --model entityonly
python per_code_metrics.py --model fulltext
```

Each call reads `models/<model>/predictions_test.feather`, applies the paper-tuned decision boundary, counts training frequencies from `data/processed/mimiciv_icd10/entity-only/train.parquet`, joins long titles from `data/code_descriptions/d_icd_{diagnoses,procedures}.csv` (filtered to ICD-10), and overwrites the supplementary CSV in place. The feather inputs come from the same `--models entity-only,fulltext` download as Table 7. Pass `--predictions <path>`, `--threshold <float>`, `--train-parquet <path>`, or `--output <path>` to point at other artefacts.

A note on `train_frequency`: upstream Edin et al. dataset prep (`explainable_medical_coding/data/make_mimiciv_icd10.py:29`) calls `remove_rare_codes(..., min_count=10)` on the _combined_ pre-split data, so every code in the model's label space has at least 10 occurrences across train+val+test. The `train_frequency` column in the per-code CSVs counts only the train split, so values can drop below 10 (about 15% of codes do) when most of a code's occurrences happen to land in val/test. These rare-in-train codes were still part of training and do produce successful predictions, so low `train_frequency` rows are expected.

## Tables 10 and 11 ablation recipes

**Table 10 (entity-token and ordering ablations).** Three of the four Table 10 ablations are opt-in flags on `ner/create_train_input.py`:

- _No entity-type tokens_: `--remove_tokens` strips `<disorder>`, `<medication>`, `<procedure>`, `<health_context>`, `<abnormal_finding>` from each entity span.
- _Plain-text category indicators_: `--replace_tokens` replaces those tokens with plain-text labels (`<disorder>` becomes `(Disorder)`).
- _Randomised entity / heading order_: `--shuffle`.

The "no AC" row is the fourth ablation. It has no shipped flag because assertion filtering is hard-coded inside `ner/extract_entities.py`. Reproducing it requires patching the AC filter logic locally rather than toggling a CLI option.

**Table 11 (entity-type subsets).** No shipped flag for this; the manual recipe is:

1. Run `ner/extract_entities.py` once on the full corpus to produce the per-entity CSV.
2. Pre-filter that CSV to only the entity types you want. Current `extract_entities.py` outputs the type as the leading tag in `text`, so filter on tag prefixes such as `<disorder>` and `<procedure>`; if you are working from annotation CSVs that already have `entity_label`, filter that column instead.
3. Pass the filtered CSV to `ner/create_train_input.py` and continue with the standard PLM-CA training command.

## Code evidence scoring quick guide

Use [`docs/evidence.md`](evidence.md) for the full MDACE recipe. The short version is that code evidence scoring is a separate MDACE evaluation flow, not the same thing as the Table 7 ICD prediction feathers. It requires MDACE plus the credentialed MIMIC-III note text used by MDACE, and for the entity-only arm it also runs NER + AC over the MDACE discharge-summary subset.

For the full-text model, scoring has three steps:

1. Run `infer_with_explanations_fulltext.py` on the MDACE-DC `val`, `train`, and `test` CSVs at the permissive `1e-6` attribution threshold. This writes per-word candidate evidence.
2. Tune the attribution threshold on the `val` per-word output in `code_evidence/code_evidence_eval.ipynb` Stage 2. The manuscript value is `0.0013877551020408164`.
3. Run `merge_contiguous_spans.py` on the `train` and `test` per-word outputs with the val-derived threshold, then use the notebook's "Get Basic Metrics" cells on the merged train+test output. That is the span-level scoring path for Table 8; Stage 4 computes the Appendix B / Table 12 MDACE classification metrics.

`code_evidence/visualise_predictions_explanations.py` is wired to the entity-only evidence output only. It is useful for per-note HTML review, but it is not the full-text MDACE scoring path.

## Tuned thresholds

Attribution thresholds were tuned on the MDACE validation split by maximising partial-match F2 (Stage 2 of the eval notebook). The probability thresholds are the MIMIC-IV decision boundaries reused for MDACE evaluation; they also equal the defaults of `--decision-boundary` on the inference scripts and `PROBABILITY_THRESHOLD` in `code_evidence/code_evidence_eval.ipynb`.

| Model       | Probability threshold | Attribution threshold   | Best partial-F2 (val) |
| ----------- | --------------------- | ----------------------- | --------------------- |
| Entity-only | `0.4040403962135315`  | `0.0021632653061224487` | 0.3634                |
| Full-text   | `0.4141414165496826`  | `0.0013877551020408164` | 0.4079                |

The attribution column is the scoring threshold, not necessarily the first-pass inference threshold. For entity-only evidence, `infer_with_explanations.py --attr-threshold` defaults to `0.0001` so general inference retains essentially all line-level evidence; the notebook then applies the MDACE-tuned threshold above when scoring. For full-text evidence, `infer_with_explanations_fulltext.py` should be run at `1e-6` for MDACE reproduction so the val sweep has the full per-word attribution distribution; pass the val-tuned value above as the third positional argument to `merge_contiguous_spans.py` before final scoring.

Re-tune locally (Stage 2) if you need a fresh sweep on different hardware or after retraining.
