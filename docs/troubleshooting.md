# Troubleshooting

What this page covers: common stumbles when running the pipeline, plus the platform- and environment-specific gotchas (Windows OpenMP, dataset caches, wandb logins) that aren't worth surfacing on the landing page but cost real time when they hit.

## Common Issues

- **`CUDA out of memory`** during entity extraction: lower `--max_workers` (default 2 in `run_pipeline.py`, 5 in direct `extract_entities.py`). Each worker loads its own NER + AC model copy.
- **Tokenizer not found** (`models/tokenizer_latest/`): the entity-only inference script and the tokenizer download share a directory name. If you skipped `--models entity-only`, both the entity-only coding model and its tokenizer are missing. Re-run with `--models entity-only`.
- **Base RoBERTa-PM not found** (`models/roberta-base-pm-m3-voc-hf`): PLM-CA inference and training need the base BioLM RoBERTa-PM encoder. From `external/plm_ca/`, run `make download_roberta`. `python data_download.py --models roberta` downloads the separate distill-align variant for NER/AC retraining and will not fix this error.
- **Missing PhysioNet files** (`mimiciv/2.2/hosp/diagnoses_icd.csv.gz` etc.): the PLM-CA `make` targets expect the full `physionet.org/files/...` directory tree from `wget --recursive`. Re-run the credentialed `wget` from `external/plm_ca/data/raw/` rather than copying individual files.
- **Wandb interactive login prompt** during PLM-CA training: append `callbacks=no_wandb` to the Hydra command for a wandb-free callbacks preset (preserves best-model + early-stopping callbacks). Alternatively set `WANDB_MODE=disabled` (or `WANDB_DISABLED=1`) before launching, or run `wandb login` once.
- **Stale HuggingFace dataset cache.** The PLM-CA loaders cache datasets under `~/.cache/huggingface/datasets/` (e.g. `mdace_inpatient_icd10`, `mimiciv_icd10`). After regenerating data via the documented `python -m explainable_medical_coding.data...` commands or the equivalent upstream `make` targets, delete the relevant cache dir if results look wrong; HuggingFace will re-bind on next run.
- **Visualisation has no entries.** The visualiser reads `results/formatted_texts/formatted_*.txt`, populated only when entity extraction is run with `--save-formatted-texts` (or via `run_pipeline.py --visualize-evidence`). Stale formatted-text files from a previous run can also leak into a fresh visualisation; clear `results/formatted_texts/` if a re-run is producing stale HTML.
- **`infer_with_explanations.py` says it can't find a CSV** when you passed a `.parquet`: this script reads CSV only. Pass the entity CSV emitted by `extract_entities.py` directly.
- **`.env` / `EXPERIMENT_PATH` errors**: the trainer falls back to `EXPERIMENT_PATH=models` when neither the environment variable nor `external/plm_ca/.env` is set. Copy `external/plm_ca/.env.example` to `external/plm_ca/.env` only if you need to redirect checkpoints.

For deeper data-prep issues, [`ac/README.md`](../ac/README.md) has a dedicated trail for the AC pipeline.

## Windows: OpenMP duplicate-init warning

When launching PLM-CA training you may see `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`. The conda env ships multiple OpenMP runtimes side-by-side (Intel via `intel-openmp`/MKL, plus `llvm-openmp` and `libgomp` pulled in by cross-channel deps); both initialisations resolve to the same DLL on this env, so it is a spurious detection. In PowerShell, set `$env:KMP_DUPLICATE_LIB_OK="TRUE"` before the command. In POSIX shells, prepend `KMP_DUPLICATE_LIB_OK=TRUE` to the command or export it first.

## Don't run make setup / make prepare_everything

Do not run `make setup` or `make prepare_everything` from the PLM-CA fork; those targets bring up a separate Poetry env that is not the supported execution path here. The `entitycoding` conda env covers every script in the repo.
