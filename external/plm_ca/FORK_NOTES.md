# Fork notes for `external/plm_ca/`

This directory is a vendored copy of
[`JoakimEdin/explainable-medical-coding`](https://github.com/JoakimEdin/explainable-medical-coding)
at commit [`8269cc7`](https://github.com/JoakimEdin/explainable-medical-coding/commit/8269cc7246b88fa5dd299191713ed7475b908537).
This file is the technical companion to the "Additions in this fork" banner at
the top of `README.md`: it lists every file we added on top of upstream, every
upstream file we modified, and the parts of upstream this fork does not
exercise. The intent is to make ours-vs-theirs unambiguous so a reader can
trust which behaviour came from where.

## Files we added

Every script below ships an argparse CLI; run it with `--help` from this
directory for full usage. The Hydra training scripts use `--help` via Hydra's
own help action.

Top-level scripts:

- `train_plm_entities.py`. Fork of upstream `train_plm.py`. Trains the
  entity-only PLM-CA checkpoint. Registers the five entity special tokens
  (`<disorder>`, `<medication>`, `<procedure>`, `<health_context>`,
  `<abnormal_finding>`) before the encoder is loaded, resizes the embedding
  table, and saves the matching tokenizer to `models/tokenizer_latest/` (the
  artefact path the inference loader and the released tokenizer agree on).
  After training, sanity-checks that the added-token embeddings actually
  changed.
- `train_plm_fulltext.py`. Fork of upstream `train_plm.py`. Trains the
  full-text PLM-CA baseline. Intentionally minimal divergence from upstream:
  no entity tokens are registered, so the upstream tokenizer is reused as-is.
  Kept as a separate file (rather than a flag on the entity script) so the
  entity-only changes stay visible alongside it.
- `infer_with_explanations.py`. Entity-side inference. Loads the entity-only
  model + `models/tokenizer_latest/`, runs PLM-CA, and emits per-line
  evidence (one line == one entity span) using AttInGrad attribution. Both
  the decision boundary and the per-token attribution cutoff are CLI flags
  (`--decision-boundary`, `--attr-threshold`). The `--decision-boundary`
  default equals the paper-tuned MDACE value for the entity-only model;
  the `--attr-threshold` default (`0.0001`) is intentionally permissive
  so general-purpose inference keeps essentially all line-level evidence.
  The tighter paper-tuned attribution cutoff used for MDACE evaluation
  is applied inside `code_evidence/code_evidence_eval.ipynb`, not as a
  CLI default here.
- `infer_with_explanations_fulltext.py`. Full-text counterpart that runs
  the full-text PLM-CA checkpoint (the manuscript's "PLM-CA (Full-text)"
  baseline, trained via `train_plm_fulltext.py`) on raw notes, aggregates
  AttInGrad subword attributions up to whole words, and emits one row per
  (note_id, code). Decision boundary exposed as `--decision-boundary`.
  Designed to be run at a permissive `attr_threshold` (default `1e-6`) so
  the threshold can be tuned post-hoc on the val per-word output without
  re-running inference.
- `merge_contiguous_spans.py`. Second stage of the full-text evidence flow.
  Reads the per-word output of `infer_with_explanations_fulltext.py` and
  collapses adjacent words into contiguous character-level spans (the unit
  MDACE annotates and the notebook evaluates against). Span attribution is
  the *mean* of the constituent word attributions. Apply the val-tuned
  attribution threshold here.
- `eval_metrics.py`. One-shot recompute of the Table 7 metrics (f1_micro,
  f1_macro, P@8, P@15, exact_match_ratio, MAP, AUC_micro, AUC_macro) from a
  predictions feather. Useful for verifying metrics on the released
  prediction feathers without re-running training or full validation.
- `per_code_metrics.py`. Per-ICD-code precision / recall / F1 / TP / FP / FN
  / `test_count` / `train_frequency` from a predictions feather. Reproduces
  the supplementary tables at `results/per_code_metrics_{entityonly,fulltext}.csv`
  and overwrites them in place by default. Codes that appear in test ground
  truth but not in the model's classifier head are emitted with tp=fp=0 and
  fn=test_count so the row is still present alongside its `train_frequency`.
- `permutation_testing.py`. Paired permutation test for Table 7 metric
  differences between two models (e.g. entity-only vs. full-text). Takes two
  prediction feathers, swaps per-note predictions on coin flips for
  `--num-permutations` iterations, and reports per-metric p-values. The
  default invocation matches the recorded paper run committed at
  `results/paper_permutation_test_results.txt`. Replaces an earlier typo'd
  `permtutation_testing.py`.
- `get_length_of_docs.py`. Computes the Table 5 word-count statistics
  (entity-only vs. full-text training parquets). Strips entity tags before
  splitting on whitespace so the comparison reflects retained source content
  rather than synthetic markup.

Configs:

- `explainable_medical_coding/config/callbacks/no_wandb.yaml`. New preset
  carrying best-model + early-stopping callbacks but no `WandbCallback`.
  Pass `callbacks=no_wandb` on the Hydra command line to disable wandb.

## Modifications to upstream files

- **`explainable_medical_coding/config/config.yaml`**. Default `data` group
  changed from `mimiciv_icd10cm` (a non-existent file) to `mimiciv_icd10` so
  bare-Hydra entrypoints (`eval_explanations`, etc.) stop crashing with
  `MissingConfigException`. The documented training commands are unaffected
  because they pass `data=mimiciv_icd10` explicitly.
- **`explainable_medical_coding/config/model/autoregressive.yaml`**.
  Hardcoded absolute Linux model path replaced with a relative path matching
  `model/plm_icd.yaml` so anyone selecting `model=autoregressive` does not
  crash on a stranger's machine path.
- **`explainable_medical_coding/trainer/callbacks.py`**. The module-import
  read of `EXPERIMENT_PATH` now uses `os.environ.get("EXPERIMENT_PATH",
  "models")` so the trainer no longer dies when `.env` is missing. The
  fallback value matches the (renamed) `.env.example`.
- **`data/raw/MDace/README.md`**. Small note added clarifying the upstream
  commit pin and CC BY 4.0 license, plus restore instructions for the
  Inpatient/Profee subdirectories. The annotation tree itself is unchanged.
- **`.gitignore`**. Added `.env` so a local override file does not leak
  into the history.
- **`.env` renamed to `.env.example`**. The committed example carries the
  documented defaults; `.env` is now gitignored. The `callbacks.py` fallback
  above keeps the documented commands working without a local `.env`.

## Training reproduction

The released entity-only and full-text checkpoints were trained with the
upstream-default training recipe and a *single* override:
`dataloader.max_batch_size=1`. The override is needed for VRAM. The upstream
`config/dataloader/defaults.yaml` keeps `batch_size: 16`, so dropping
`max_batch_size` to 1 automatically engages 16 gradient-accumulation steps
(the fallback path documented in the upstream README, and exactly the
override that README suggests for memory-constrained machines). No other
hyperparameters were touched; the manuscript's results were produced under
upstream defaults plus this one batching change.

The exact commands (run from this directory, with the `entitycoding` env
active) were:

```bash
python train_plm_entities.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10
python train_plm_fulltext.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10
```

The `data=mimiciv_icd10` override is mandatory: without it the experiment
preset silently falls back to ICD-9. Append `callbacks=no_wandb` to disable
wandb logging.

Caveat for retraining the entity-only model: `train_plm_entities.py` saves
the tokenizer to `models/tokenizer_latest/` (hardcoded, near the top of
`main()`, before any training or argument validation). Re-running it will
overwrite an existing `models/tokenizer_latest/`, including the
released-checkpoint tokenizer that `data_download.py --models entity-only`
puts there for inference. Back the released artefact up first if you need
to keep it.

Caveat for switching between the entity-only and full-text training inputs:
the upstream dataset loader
(`explainable_medical_coding/datasets/mimiciv_icd10.py`) reads from a single
hardcoded path,
`external/plm_ca/data/processed/mimiciv_icd10/{train,val,test}.parquet`. There
is no flag to point it at the entity-only or full-text variant separately,
so retraining on the other variant means physically swapping the parquets at
that location. The top-level `README.md` ("Run Training") shows one workable
layout (keep the full-text shards under `fulltext/` and the entity-only
shards under `entity-only/`, then copy whichever set you want into the top
level before training).

When you swap, also clear the HuggingFace dataset cache before retraining,
because the loader caches by dataset config name and will silently re-use
the previously cached *other* variant otherwise (silent in the sense that
the model trains, but on the wrong inputs). The cache lives at
`~/.cache/huggingface/datasets/mimiciv_icd10/` on Linux and
`%USERPROFILE%\.cache\huggingface\datasets\mimiciv_icd10\` on Windows.
Deleting that directory is safe; it will be rebuilt from the parquets on
the next run.

## Two BioLM RoBERTa-PM variants are in play

They are *not* interchangeable as initialisation weights, only as
architecture stubs (both are 12L / 768H / 50k vocab; only pre-training
weights differ).

- **`RoBERTa-base-PM-M3-Voc-hf`** (the *base* variant, from
  `https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz`)
  is what `make download_roberta` fetches and what the released entity-only
  and full-text PLM-CA checkpoints were trained from. The saved
  `model.configs.model_path` inside `models/{entityonly,fulltext}/config.yaml`
  still points at this path (`models/roberta-base-pm-m3-voc-hf`). To
  *retrain* PLM-CA from paper-faithful initialisation, you need this variant.
  `data_download.py` does not fetch it, so run `make download_roberta` (or
  download the tarball manually) before invoking
  `python train_plm_{entities,fulltext}.py`.
- **`RoBERTa-base-PM-M3-Voc-distill-align-hf`** (the distilled +
  token-aligned variant, from
  `https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz`)
  is what `data_download.py --models roberta` fetches. The NER and AC
  pipelines fine-tune from this variant. PLM-CA inference no longer falls
  back to it: `_resolve_encoder_path` in `infer_with_explanations.py` and
  `infer_with_explanations_fulltext.py` strictly requires the base variant
  at the path saved in the trained model's config and raises a
  `FileNotFoundError` pointing at `make download_roberta` if it is missing.

## Rare-code filter operates on the combined pre-split set, not per-split

Upstream `explainable_medical_coding/data/make_mimiciv_icd10.py:29` calls
`remove_rare_codes(..., min_count=10)` on the combined pre-split data,
before the train/val/test split. Empirically, every code in the released
parquets at `data/processed/mimiciv_icd10/{entity-only,fulltext}/` has at
least 10 occurrences across train+val+test combined; the minimum combined
count is exactly 10 and zero codes fall below it. The 7,942 codes in the
released models' classifier head correspond one-to-one with that filtered
combined vocabulary.

Per-split counts can still look "rare" because the filter is global: about
15% of codes in the entity-only `train.parquet` have a *train-only* count
below 10 (down to 1), simply because most of their 10+ combined occurrences
landed in val or test rather than train. These are genuine training-time
codes and the released models do produce successful predictions on them, so
the low `train_frequency` rows in `results/per_code_metrics_*.csv` are
expected, not a leakage or filter-bypass signal.

## Things upstream does that this fork does not use

- `make prepare_everything` and the Poetry-based setup are not the supported
  execution path here; the parent repo ships a single `entitycoding` conda
  env that satisfies both the fork and the added scripts.
- Adversarial training presets (IGR / TM / PGD) and LIME / KernelSHAP
  explainers are upstream-only and not exercised by the entity-coding flow.
- Upstream `eval_explanations.py` is kept in tree for reference but is
  not part of the documented reproduction path; use the fork's
  `eval_metrics.py` / `permutation_testing.py` instead. Upstream
  `train_plm.py` is not vendored here; the fork's
  `train_plm_{entities,fulltext}.py` pair are the supported training
  entrypoints.

## How to refresh from upstream

If you ever need to resync to a newer upstream commit:

1. Apply the modifications above on top of the new tree (or rebase the
   above changes; they are all small).
2. Verify the tokenizer save/load path in `train_plm_entities.py` still
   matches the distributed artefact name (currently `models/tokenizer_latest/`).
3. Confirm `data: mimiciv_icd10` still resolves to a real config under
   `explainable_medical_coding/config/data/`.
4. Smoke-test `--help` on every added top-level script
   (`infer_with_explanations.py`, `infer_with_explanations_fulltext.py`,
   `merge_contiguous_spans.py`, `eval_metrics.py`, `per_code_metrics.py`,
   `permutation_testing.py`, `get_length_of_docs.py`) so any drift in the
   argparse layout, default values, or upstream import paths surfaces
   before you run anything heavy.
5. Re-confirm `_resolve_encoder_path` still strictly requires the base
   PM-RoBERTa variant; if upstream changes how `model.configs.model_path`
   is stored, the resolver may need to be updated.
