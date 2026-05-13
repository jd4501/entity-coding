# Inter-Annotator Agreement (IAA) Package

Reproduces the agreement numbers reported in the paper:

> "a second author annotated a random subset of documents containing roughly 1500 entities [...] exact-span F1 and Cohen's Kappa, yielding scores of 0.77 and 0.81, respectively."

## Contents

| File                       | Tracked?           | Role                                                                           |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------ |
| `compute_iaa.py`           | yes                | Scoring script (token-level Kappa + F1, entity-level exact/partial F1)         |
| `README.md`                | yes                | This file                                                                      |
| `annotations/texts.txt`    | no (MIMIC-derived) | 8 source documents (de-identified discharge summaries), separated by `#######` |
| `annotations/author1.json` | no (MIMIC-derived) | Annotations by author 1 (the primary annotator): 8 docs, 1486 entities         |
| `annotations/author2.json` | no (MIMIC-derived) | Annotations by author 2: 8 docs, 1435 entities                                 |

The text and JSON annotation files contain MIMIC-IV-Note-derived material and
are therefore subject to PhysioNet credentialed-access terms. They are not
committed here. Place them into
`annotations/` before running `compute_iaa.py`.

Each annotation JSON is a list of 8 documents; each document is a list of
entity objects with `start`, `end`, and `labels` fields, where offsets index into
the corresponding document in the matching `annotations/texts*.txt` file.

## Requirements

- Python 3.x
- `scikit-learn`

## Reproducing the paper's numbers

```bash
python compute_iaa.py annotations/texts.txt annotations/author1.json annotations/author2.json -d '#######'
```

Expected output (excerpt):

```text
Token-Level Cohen's Kappa: 0.8095
Entity-Level Exact Match Metrics:
  Precision: 0.7791, Recall: 0.7524, F1: 0.7655
Entity-Level Partial Match Metrics:
  Precision: 0.8760, Recall: 0.8459, F1: 0.8607
Total entities: gold=1486, pred=1435
```

Rounded: exact-span F1 = **0.77**, Cohen's Kappa = **0.81**.

## Running on the subsets

```bash
python compute_iaa.py annotations/texts-set1.txt annotations/author1-set1.json annotations/author2-set1.json -d '#######'
python compute_iaa.py annotations/texts-set2.txt annotations/author1-set2.json annotations/author2-set2.json -d '#######'
```

## Notes on scoring

- **Token-level labels** are assigned by tokenizing on whitespace (`\S+`) and
  giving each token the label of the annotation whose span covers the token's
  midpoint (else `O`).
- **Cohen's Kappa** is computed over all tokens.
- **Token-level F1** (macro/micro + per-class report) is computed after filtering
  out tokens where both annotators labeled `O`, to avoid inflation from the
  large `O` majority.
- **Entity-level exact match**: same `start`, `end`, and label. Greedy 1-to-1.
- **Entity-level partial match**: any span overlap > 0 and same label. Greedy
  1-to-1.
- Author 1 (the primary annotator) is treated as gold for precision/recall;
  swapping arguments swaps precision and recall but leaves F1 and Kappa
  unchanged.

## Label set

`abnormal_finding`, `disorder`, `health_context`, `medication`, `normal_finding`,
`procedure` (plus `O` for tokens outside any entity).
