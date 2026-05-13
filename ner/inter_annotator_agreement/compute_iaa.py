"""Compute inter-annotator agreement (IAA) for the entity annotations.

Reproduces the agreement numbers reported in Section 3.1 of the paper (exact-span
F1 = 0.77, Cohen's Kappa = 0.81), using a second annotator's labels on the
same documents as the primary annotator.

Inputs (one matched triple per run):
  - text file from ``annotations/`` with the source documents the two
    annotators saw. It can be a JSON array of strings, or a plain text file
    with documents separated by a literal delimiter (passed via
    ``--delimiter``).
  - two annotation JSON files, one per annotator. Each is a JSON list with
    one element per document; each element is a list of entity objects with
    ``start``, ``end``, and ``labels`` fields, where offsets index into the
    matching document in the text file.

Outputs (printed to stdout, the script's intentional reporting interface):
  - token-level Cohen's Kappa over all tokens.
  - token-level macro/micro F1 and per-class report after dropping tokens
    where both annotators chose the outside label ``O``. Filtering avoids
    inflation from the dominant ``O`` class.
  - entity-level precision / recall / F1 under exact-span and partial-span
    matching. The first annotator is treated as gold for precision/recall;
    swapping the two arguments swaps precision and recall but leaves F1 and
    Kappa unchanged. Author 1 is the primary annotator.

Key flags:
  --delimiter / -d  Plain-text document separator (e.g. ``#######``). Omit
                    when the text file is a JSON array.
  --max-debug-examples  Cap on per-token debug listings shown at DEBUG level.
  --log-level       Logging verbosity for milestones and debug listings.
                    Final metric numbers are always printed via stdout.

Example (paper-equivalent invocation):
    python ner/inter_annotator_agreement/compute_iaa.py \\
        ner/inter_annotator_agreement/annotations/texts.txt \\
        ner/inter_annotator_agreement/annotations/author1.json \\
        ner/inter_annotator_agreement/annotations/author2.json \\
        -d '#######'
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

LOGGER = logging.getLogger("compute_iaa")

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = SCRIPT_DIR / "annotations"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
OUTSIDE_LABEL = "O"
TOKEN_RE = re.compile(r"\S+")


def _configure_stdio() -> None:
    """Use UTF-8 streams on Windows terminals when Python exposes reconfigure()."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_texts(path: Path, delimiter: str | None = None) -> list[str]:
    """Load source documents.

    With ``delimiter`` set, the file is read as plain text and split on the
    literal delimiter. Otherwise it must be a JSON array of strings.
    """
    if delimiter is None:
        with path.open("r", encoding="utf-8") as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            raise ValueError(
                f"{path}: expected a JSON array of documents (or pass --delimiter "
                f"to read a plain-text file)."
            )
    else:
        content = path.read_text(encoding="utf-8")
        texts = [doc.strip() for doc in content.split(delimiter) if doc.strip()]
    return texts


def load_annotations(path: Path) -> list[list[dict]]:
    """Load a per-document annotation list from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    if not isinstance(annotations, list):
        raise ValueError(f"{path}: expected a JSON array (one entry per document).")
    return annotations


def resolve_input_path(path: Path) -> Path:
    """Resolve moved text inputs from the script's annotations directory."""
    if path.exists():
        return path
    if path.suffix.lower() == ".txt":
        candidate = ANNOTATIONS_DIR / path.name
        if candidate.exists():
            return candidate
    return path


def tokenize(text: str) -> list[dict]:
    """Whitespace-tokenize and record character offsets and token midpoint."""
    tokens = []
    for match in TOKEN_RE.finditer(text):
        start, end = match.start(), match.end()
        tokens.append(
            {
                "text": match.group(),
                "start": start,
                "end": end,
                # Midpoint is used to assign each token to at most one annotation
                # span, which keeps the token-label sequences aligned across the
                # two annotators even when their span boundaries differ slightly.
                "center": (start + end) / 2,
            }
        )
    return tokens


def assign_labels(tokens: list[dict], annotations: list[dict]) -> list[str]:
    """Assign each token the label of the annotation whose span covers its midpoint."""
    labels = [OUTSIDE_LABEL] * len(tokens)
    for annotation in annotations:
        ann_start = annotation["start"]
        ann_end = annotation["end"]
        ann_label = annotation["labels"][0] if annotation.get("labels") else OUTSIDE_LABEL
        for i, token in enumerate(tokens):
            if ann_start <= token["center"] < ann_end:
                labels[i] = ann_label
    return labels


def filter_non_outside_pairs(
    labels_a: list[str], labels_b: list[str]
) -> tuple[list[str], list[str]]:
    """Drop tokens where both annotators chose the outside label.

    The outside class dominates token counts, so leaving it in inflates F1.
    """
    filtered_a, filtered_b = [], []
    for la, lb in zip(labels_a, labels_b):
        if not (la == OUTSIDE_LABEL and lb == OUTSIDE_LABEL):
            filtered_a.append(la)
            filtered_b.append(lb)
    return filtered_a, filtered_b


def log_token_debug(
    tokens: list[dict],
    labels_a: list[str],
    labels_b: list[str],
    max_examples: int = 40,
) -> None:
    """Emit per-token match/mismatch examples at DEBUG level."""
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    matches, non_matches = [], []
    for token, la, lb in zip(tokens, labels_a, labels_b):
        if la == lb:
            matches.append((token, la))
        else:
            non_matches.append((token, la, lb))

    LOGGER.debug("Total tokens: %d", len(tokens))
    LOGGER.debug("Matching token labels: %d", len(matches))
    LOGGER.debug("Non-matching token labels: %d", len(non_matches))

    LOGGER.debug("Example matching tokens (up to %d):", max_examples)
    for token, label in matches[:max_examples]:
        LOGGER.debug(
            "  '%s' (span %d-%d) -> %s",
            token["text"], token["start"], token["end"], label,
        )

    LOGGER.debug("Example non-matching tokens (up to %d):", max_examples)
    for token, la, lb in non_matches[:max_examples]:
        LOGGER.debug(
            "  '%s' (span %d-%d) annotator1=%s annotator2=%s",
            token["text"], token["start"], token["end"], la, lb,
        )


def _entity_label(entity: dict) -> str:
    return entity["labels"][0] if entity.get("labels") else OUTSIDE_LABEL


def match_exact(gold: list[dict], pred: list[dict]) -> int:
    """Count entities with identical start, end, and label.

    Greedy one-to-one matching: each gold entity may pair with at most one
    predicted entity. This matches the manuscript's reporting convention.
    """
    pred_matched = [False] * len(pred)
    matches = 0
    for g in gold:
        g_label = _entity_label(g)
        for j, p in enumerate(pred):
            if pred_matched[j]:
                continue
            if g["start"] == p["start"] and g["end"] == p["end"] and g_label == _entity_label(p):
                matches += 1
                pred_matched[j] = True
                break
    return matches


def match_partial(gold: list[dict], pred: list[dict]) -> int:
    """Count entities with any positive span overlap and the same label.

    Greedy one-to-one matching, same convention as :func:`match_exact`.
    """
    pred_matched = [False] * len(pred)
    matches = 0
    for g in gold:
        g_label = _entity_label(g)
        for j, p in enumerate(pred):
            if pred_matched[j] or _entity_label(p) != g_label:
                continue
            overlap = min(g["end"], p["end"]) - max(g["start"], p["start"])
            if overlap > 0:
                matches += 1
                pred_matched[j] = True
                break
    return matches


def _prf(total_tp: int, total_gold: int, total_pred: int) -> tuple[float, float, float]:
    fp = total_pred - total_tp
    fn = total_gold - total_tp
    precision = total_tp / (total_tp + fp) if (total_tp + fp) > 0 else 0.0
    recall = total_tp / (total_tp + fn) if (total_tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_entity_metrics(
    annotations_gold: list[list[dict]],
    annotations_pred: list[list[dict]],
    matcher,
) -> tuple[float, float, float, int, int]:
    """Aggregate per-document TP/gold/pred counts under the given matcher."""
    total_tp = total_gold = total_pred = 0
    for gold, pred in zip(annotations_gold, annotations_pred):
        total_tp += matcher(gold, pred)
        total_gold += len(gold)
        total_pred += len(pred)
    precision, recall, f1 = _prf(total_tp, total_gold, total_pred)
    return precision, recall, f1, total_gold, total_pred


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text_file",
        type=Path,
        help=(
            "Source documents, typically annotations/texts.txt "
            "(JSON array of strings, or plain text with --delimiter)."
        ),
    )
    parser.add_argument(
        "annotations_file1",
        type=Path,
        help="Annotator 1 JSON (treated as gold for precision/recall).",
    )
    parser.add_argument(
        "annotations_file2",
        type=Path,
        help="Annotator 2 JSON.",
    )
    parser.add_argument(
        "--delimiter", "-d",
        default=None,
        help="Document separator for plain-text input (omit for JSON input).",
    )
    parser.add_argument(
        "--max-debug-examples",
        type=int,
        default=40,
        help="Per-token debug examples to print at DEBUG level (default: 40).",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity for progress and debug listings (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    args.text_file = resolve_input_path(args.text_file)

    for label, path in [
        ("text file", args.text_file),
        ("annotations file 1", args.annotations_file1),
        ("annotations file 2", args.annotations_file2),
    ]:
        if not path.exists():
            LOGGER.error("%s not found: %s", label, path)
            raise SystemExit(2)

    try:
        LOGGER.info("Loading documents from %s", args.text_file)
        texts = load_texts(args.text_file, delimiter=args.delimiter)
        LOGGER.info("Loading annotator 1 from %s", args.annotations_file1)
        annotations_docs1 = load_annotations(args.annotations_file1)
        LOGGER.info("Loading annotator 2 from %s", args.annotations_file2)
        annotations_docs2 = load_annotations(args.annotations_file2)
    except (ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)

    if not (len(texts) == len(annotations_docs1) == len(annotations_docs2)):
        LOGGER.error(
            "Document count mismatch: texts=%d, annotator1=%d, annotator2=%d",
            len(texts), len(annotations_docs1), len(annotations_docs2),
        )
        raise SystemExit(2)

    LOGGER.info("Loaded %d documents", len(texts))

    all_tokens: list[dict] = []
    all_labels1: list[str] = []
    all_labels2: list[str] = []
    for text, ann1, ann2 in zip(texts, annotations_docs1, annotations_docs2):
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        all_labels1.extend(assign_labels(tokens, ann1))
        all_labels2.extend(assign_labels(tokens, ann2))

    log_token_debug(all_tokens, all_labels1, all_labels2, max_examples=args.max_debug_examples)

    kappa = cohen_kappa_score(all_labels1, all_labels2)
    print(f"Token-Level Cohen's Kappa: {kappa:.4f}")

    filtered_labels1, filtered_labels2 = filter_non_outside_pairs(all_labels1, all_labels2)
    if not filtered_labels1:
        print("\nNo tokens available for F1 evaluation after filtering out 'O' cases.")
    else:
        f1_macro = f1_score(filtered_labels1, filtered_labels2, average="macro", zero_division=0)
        f1_micro = f1_score(filtered_labels1, filtered_labels2, average="micro", zero_division=0)
        report = classification_report(filtered_labels1, filtered_labels2, zero_division=0)
        print(f"\nToken-Level Macro F1 Score (excluding both 'O'): {f1_macro:.4f}")
        print(f"Token-Level Micro F1 Score (excluding both 'O'): {f1_micro:.4f}")
        print("\nToken-Level Classification Report (excluding both 'O'):")
        print(report)

    exact_p, exact_r, exact_f1, total_gold, total_pred = compute_entity_metrics(
        annotations_docs1, annotations_docs2, match_exact
    )
    partial_p, partial_r, partial_f1, _, _ = compute_entity_metrics(
        annotations_docs1, annotations_docs2, match_partial
    )

    print("\nEntity-Level Exact Match Metrics:")
    print(f"  Precision: {exact_p:.4f}, Recall: {exact_r:.4f}, F1: {exact_f1:.4f}")
    print("\nEntity-Level Partial Match Metrics:")
    print(f"  Precision: {partial_p:.4f}, Recall: {partial_r:.4f}, F1: {partial_f1:.4f}")
    print(f"\nTotal entities: gold={total_gold}, pred={total_pred}")


if __name__ == "__main__":
    main()
