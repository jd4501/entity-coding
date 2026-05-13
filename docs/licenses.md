# Licenses and citations

What this page covers: the per-source license terms that apply to each piece of this repository (code, model weights, source clinical data, annotation releases, synthetic samples), and the secondary citations a publication building on this work should include alongside the main paper. The root README has the high-level summary; this page is the authoritative breakdown.

## Code

The source code in this repository is released under the [MIT License](../LICENSE).

`external/plm_ca/` is third-party code, vendored for convenience. It is a copy of [JoakimEdin/explainable-medical-coding](https://github.com/JoakimEdin/explainable-medical-coding) at the commit linked in the overview, included here only to make pipeline setup effortless and to pin the exact upstream version this work was developed against. It carries its own MIT license (`external/plm_ca/LICENSE`, (c) 2023 Joakim Edin). The handful of new scripts added on top of the vendored tree (listed in [`external/plm_ca/README.md`](../external/plm_ca/README.md) under "Additions in this fork") are MIT under this repository's root LICENSE; any other modifications are minor adaptations of upstream code. Any work that uses or builds on code under `external/plm_ca/` must cite Edin et al. (2024); see [Citations](#citations).

## Trained model weights (non-commercial only)

The NER, AC, entity-only ICD-10 coding, and full-text ICD-10 coding model weights distributed via `data_download.py` were trained on data subject to non-commercial licenses (MIMIC-III, MIMIC-IV, MIMIC-IV-Note, and the i2b2/n2c2 challenge corpora for AC). Use of these weights for commercial purposes is forbidden. They are also not released for clinical decision support or billing automation. This restriction is inherited from the underlying training data and applies regardless of the MIT license that covers the code that produced them. The same constraint applies to the BioLM RoBERTa-PM encoder weights used as initialisation (Lewis et al. 2020; full BibTeX in [Citations](#citations)). If you need a commercially-usable model, retrain on data licensed for that use.

## Source clinical data (PhysioNet credentialed access)

MIMIC-III, MIMIC-IV, and MIMIC-IV-Note are distributed by [PhysioNet](https://physionet.org/) under the [PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/) and require a PhysioNet account, completion of a free human-subjects research training course (CITI "Data or Specimens Only Research"), and a signed Data Use Agreement. The planned MIMIC-IV-Ext-EntityCoding annotation release will follow the same access model. This repository does not redistribute any MIMIC data; users must obtain it from PhysioNet directly. See [`data/README.md`](../data/README.md) and [`external/plm_ca/data/raw/MDace/README.md`](../external/plm_ca/data/raw/MDace/README.md) for the expected layout once credentialing is complete.

i2b2/n2c2 challenge data (used to train the AC model) requires registration with the [DBMI portal](https://portal.dbmi.hms.harvard.edu/) and a separate signed DUA; see [`ac/README.md`](../ac/README.md).

## MIMIC-III bvanaken assertion labels

The AC model is additionally trained on the public clinical-assertion label CSVs from [bvanaken/clinical-assertion-data](https://github.com/bvanaken/clinical-assertion-data) (van Aken et al. 2021). The label files are downloadable from the public GitHub repo, but they index MIMIC-III note text and only become useful when joined to `NOTEEVENTS.csv` from MIMIC-III v1.4, which remains under PhysioNet credentialed access. See [`ac/README.md`](../ac/README.md) for the staging layout. If you use these labels, please cite van Aken et al. (2021); BibTeX is in [Citations](#citations).

## MIMIC-IV-Ext-EntityCoding annotations

The 400-note annotation release accompanying the paper is intended for PhysioNet distribution under the same credentialed-access terms as the underlying MIMIC-IV-Note text. Until that project is published, it is not inspectable from this repository. If you use the dataset after publication, please cite the paper and include the PhysioNet release in your data availability statement.

## MDACE annotations

The MDACE annotation tree under `external/plm_ca/data/raw/MDace/` was vendored by the upstream PLM-CA repository, not by this project. MDACE was created by Cheng et al. and is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) at [3mcloud/MDACE](https://github.com/3mcloud/MDACE). The note text it annotates still comes from MIMIC-III and is subject to the PhysioNet terms above. If you use these annotations, please cite Cheng et al. (2023); BibTeX is in [Citations](#citations).

## Synthetic sample notes

`data/sample_data/sample_notes.csv` contains GPT-4o-generated synthetic clinical notes authored for this project. They contain no real patient data and are released under the same MIT license as the code.

## Citations

The main paper BibTeX lives in the root [README](../README.md#how-to-cite). The entries below are the secondary citations a publication building on this work should include alongside it.

For the annotation dataset (exact citation pending PhysioNet publication):

```bibtex
@article{unpublished,
    author = {Douglas, James C and Gan, Yidong and Hachey, Ben and Kummerfeld, Jonathan},
    title = {{MIMIC-IV-Ext-EntityCoding: Clinical Named Entity Recognition and Assertion Classification Dataset for ICD Coding}},
    journal = {{PhysioNet}},
    year = {2026},
    month = may,
    note = {Version 1.0.0; PhysioNet publication pending}
}
```

Any work using code under `external/plm_ca/` must additionally cite Edin et al. (2024):

```bibtex
@inproceedings{edin-etal-2024-unsupervised,
    title = "An Unsupervised Approach to Achieve Supervised-Level Explainability in Healthcare Records",
    author = "Edin, Joakim  and
      Maistro, Maria  and
      Maal{\o}e, Lars  and
      Borgholt, Lasse  and
      Havtorn, Jakob Drachmann  and
      Ruotsalo, Tuukka",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.280/",
    doi = "10.18653/v1/2024.emnlp-main.280",
    pages = "4869--4890"
}
```

If you use the AC model or its training data, please also cite the bvanaken assertion labels:

```bibtex
@inproceedings{van-aken-etal-2021-assertion,
    title = "Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?",
    author = "van Aken, Betty  and
      Trajanovska, Ivana  and
      Siu, Amy  and
      Mayrdorfer, Manuel  and
      Budde, Klemens  and
      Loeser, Alexander",
    editor = "Shivade, Chaitanya  and
      Gangadharaiah, Rashmi  and
      Gella, Spandana  and
      Konam, Sandeep  and
      Yuan, Shaoqing  and
      Zhang, Yi  and
      Bhatia, Parminder  and
      Wallace, Byron",
    booktitle = "Proceedings of the Second Workshop on Natural Language Processing for Medical Conversations",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nlpmc-1.5/",
    doi = "10.18653/v1/2021.nlpmc-1.5",
    pages = "35--40"
}
```

If you use any of the trained model checkpoints distributed with this project, please also cite the BioLM RoBERTa-PM encoder used as initialisation:

```bibtex
@inproceedings{lewis-etal-2020-pretrained,
    title = "Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art",
    author = "Lewis, Patrick  and
      Ott, Myle  and
      Du, Jingfei  and
      Stoyanov, Veselin",
    editor = "Rumshisky, Anna  and
      Roberts, Kirk  and
      Bethard, Steven  and
      Naumann, Tristan",
    booktitle = "Proceedings of the 3rd Clinical Natural Language Processing Workshop",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.clinicalnlp-1.17/",
    doi = "10.18653/v1/2020.clinicalnlp-1.17",
    pages = "146--157"
}
```

If you use the MDACE annotations:

```bibtex
@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and Jafari, Rana  and Russell, April  and Klopfer, Russell  and Lu, Edmond  and Striner, Benjamin  and Gormley, Matthew",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416",
    doi = "10.18653/v1/2023.acl-long.416",
    pages = "7534--7550"
}
```
