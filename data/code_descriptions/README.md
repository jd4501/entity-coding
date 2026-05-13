# ICD Code Descriptions

Long-title lookup tables for ICD-9 and ICD-10 codes, used to gloss model
predictions with human-readable descriptions.

## Files

| File                  | Rows    | Schema                              | Notes                                                                                       |
| --------------------- | ------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `d_icd_diagnoses.csv` | 109,775 | `icd_code,icd_version,long_title`   | Both ICD-9-CM and ICD-10-CM diagnosis codes; `icd_version` is `9` or `10`.                  |
| `d_icd_procedures.csv`| 85,257  | `icd_code,icd_version,long_title`   | Both ICD-9-CM and ICD-10-PCS procedure codes; `icd_version` is `9` or `10`.                 |

## Provenance

Snapshotted from MIMIC-IV v2.2
(`physionet.org/files/mimiciv/2.2/hosp/d_icd_{diagnoses,procedures}.csv.gz`),
matching the MIMIC-IV-Note v2.2 release that the ICD coding models in this
project were trained against. The vendored PLM-CA fork resolves the same files
out of `external/plm_ca/data/raw/physionet.org/files/mimiciv/2.2/hosp/`.

The exact ICD-10-CM/PCS revision underneath these tables is not pinned; verify
against the latest CMS release if a code lookup looks stale.

## Consumers

- `code_evidence/visualise_predictions_explanations.py` joins predicted codes
  to long titles when rendering per-note HTML.
- `external/plm_ca/per_code_metrics.py` joins long titles into the
  supplementary per-code metrics CSVs at
  `results/per_code_metrics_{entityonly,fulltext}.csv`, filtered to ICD-10.
