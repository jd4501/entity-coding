# Sample notes

Synthetic discharge summaries used for smoke testing the pipeline without
needing PhysioNet credentialed access.

## Files

| File | What | Notes |
|---|---|---|
| `sample_notes.csv` | Three GPT-4o-generated discharge-summary-style notes | Columns: `note_id`, `text`. Safe to commit. |

## Provenance and intended use

These notes were authored for this project by prompting GPT-4o for
realistic-looking but fictional discharge summaries. They contain no real
patient data and are released under the same MIT license as the code in this
repository. The texts are deliberately short and stylised; they are intended
for verifying the entity-extraction + ICD-coding pipeline runs end-to-end on a
fresh checkout, not for benchmarking model quality.

For meaningful evaluation, use the credentialed datasets described in
[docs/inference.md](../../docs/inference.md#what-data-do-i-need) and
`data/README.md`.

## Quick start

From the repo root, with the `entitycoding` conda env active:

```bash
python run_pipeline.py data/sample_data/sample_notes.csv --visualize-entities --visualize-evidence
```

Reference outputs from the same input live under
`results/sample_results/` so you can sanity-check a fresh run against a
committed baseline.
