# MIMIC-IV-Note Local Staging

Put ad hoc MIMIC-IV or MIMIC-IV-Note files used by the top-level NER/data
preparation helpers here, for example:

- `discharge.csv`
- `services.csv`
- cleaned or sampled note files used outside the PLM-CA Makefile flow

Source projects:

- MIMIC-IV-Note v2.2: <https://physionet.org/content/mimic-iv-note/2.2/>
- MIMIC-IV v2.2: <https://physionet.org/content/mimiciv/2.2/>

For this local staging directory, it is fine to place decompressed CSVs such as
`discharge.csv` and `services.csv`. For PLM-CA processing, keep the compressed
PhysioNet recursive-download tree under `external/plm_ca/data/raw/` instead.

Do not use this directory for PLM-CA's own downloads. Keep those under
`external/plm_ca/data/` so the vendored PLM-CA tools continue to work.
