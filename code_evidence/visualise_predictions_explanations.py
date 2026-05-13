"""HTML visualisation of ICD code predictions and their entity evidence.

Renders one self-contained HTML page per note. Each page shows the cleaned
note text on the left and a list of predicted ICD-10 codes on the right.
Selecting a code highlights the entity spans the model attributed weight to,
and two sliders control the minimum prediction probability and minimum
evidence attribution shown.

This script visualises the entity-only evidence path only: the predicted
codes and per-line attributions emitted by
external/plm_ca/infer_with_explanations.py, where each "line" is one entity.
The full-text evidence flow (infer_with_explanations_fulltext.py +
merge_contiguous_spans.py, with character-level spans over the raw note) is
not supported by this visualiser.

Run first, in order:
  1. ner/extract_entities.py <input_notes> --save-formatted-texts
     (produces results/formatted_texts/formatted_<note_id>.txt and the
     entities CSV consumed by step 2).
  2. external/plm_ca/infer_with_explanations.py
     (produces the inference CSV consumed by this script).
  run_pipeline.py with --visualize-evidence chains all three steps.

Inputs:
  * Inference CSV from external/plm_ca/infer_with_explanations.py with the
    columns note_id, predicted_code, predicted_code_probability,
    evidence_line_numbers, evidence_spans, evidence_texts,
    evidence_attributions.
  * Formatted note bodies under the directory passed as --formatted-dir
    (default: results/formatted_texts/), produced by step 1 above.
  * data/code_descriptions/d_icd_{diagnoses,procedures}.csv for the ICD long
    titles shown in the sidebar.
  * ner/section_patterns.json for the section-header literals that get bolded
    inside the rendered note.

Outputs:
  One <note_id>.html file per note found in both the inference CSV and the
  formatted-texts directory, written into the directory passed as
  --output-dir (default: results/visualised_notes/).

Example:
    python code_evidence/visualise_predictions_explanations.py \\
        results/coded/sample_notes_results.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger("visualise_predictions_explanations")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_FORMATTED_DIR = REPO_ROOT / "results" / "formatted_texts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "visualised_notes"
SECTION_PATTERNS_FILE = REPO_ROOT / "ner" / "section_patterns.json"
DIAGNOSES_FILE = REPO_ROOT / "data" / "code_descriptions" / "d_icd_diagnoses.csv"
PROCEDURES_FILE = REPO_ROOT / "data" / "code_descriptions" / "d_icd_procedures.csv"

REQUIRED_INFERENCE_COLUMNS = (
    "note_id",
    "predicted_code",
    "predicted_code_probability",
    "evidence_spans",
    "evidence_attributions",
)

# Use UTF-8 stdio so unicode prints render on Windows (cp1252) too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def normalize_icd_code(code: str) -> str:
    """Strip any decimal point and uppercase, so 'I10.9' and 'i109' match."""
    return code.replace(".", "").upper()


def format_icd_code(code: str) -> str:
    """Re-insert the decimal point after the third character for display."""
    if len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


def load_section_literals(path: Path) -> set[str]:
    """Load section-header literals from section_patterns.json (lowercased)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["literal"].lower() for item in data.get("section_rules", [])}


def load_icd_descriptions() -> dict[str, dict[str, str]]:
    """Build a {normalized_code: {title, type, version}} map from the two CMS
    descriptor CSVs shipped under data/code_descriptions/. ICD-9 and ICD-10
    rows are kept; later versions are skipped."""
    descriptions: dict[str, dict[str, str]] = {}
    for path, code_type in ((DIAGNOSES_FILE, "diagnosis"), (PROCEDURES_FILE, "procedure")):
        if not path.exists():
            LOGGER.warning("ICD %s descriptions file not found: %s", code_type, path)
            continue
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                version = row["icd_version"]
                if version not in ("9", "10"):
                    continue
                descriptions[normalize_icd_code(row["icd_code"])] = {
                    "title": row["long_title"],
                    "type": code_type,
                    "version": version,
                }
    return descriptions


_BOLD_PREFIX = "<u><strong>"
_BOLD_SUFFIX = "</strong></u>"


def bold_literals_in_text(text: str, literals: set[str]) -> tuple[str, list[tuple[int, int]]]:
    """Wrap each (case-insensitive) section-header literal in <u><strong> tags.

    Returns the updated text plus a flat list of (insertion_position,
    insertion_length) pairs describing each prefix and suffix tag inserted
    into the original text. :func:`adjust_evidence_spans` consumes that list
    to translate evidence-span indices from original-text coordinates to
    bolded-text coordinates, including for spans that overlap a bolded header.
    """
    matches: list[tuple[int, int]] = []
    for literal in literals:
        for match in re.finditer(re.escape(literal), text, re.IGNORECASE):
            matches.append((match.start(), match.end()))

    # Greedy non-overlap, preferring the longest match starting at each point.
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    non_overlapping: list[tuple[int, int]] = []
    last_end = -1
    for start, end in matches:
        if start >= last_end:
            non_overlapping.append((start, end))
            last_end = end

    pieces: list[str] = []
    shifts: list[tuple[int, int]] = []
    last_index = 0
    for start, end in non_overlapping:
        pieces.append(text[last_index:start])
        pieces.append(_BOLD_PREFIX)
        pieces.append(text[start:end])
        pieces.append(_BOLD_SUFFIX)
        shifts.append((start, len(_BOLD_PREFIX)))
        shifts.append((end, len(_BOLD_SUFFIX)))
        last_index = end
    pieces.append(text[last_index:])
    return "".join(pieces), shifts


def adjust_evidence_spans(
    spans: list[tuple[int, int]],
    shifts: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map original-text spans to bolded-text spans.

    Each `(pos, amount)` shift represents an insertion of `amount` characters
    at position `pos` in the original text. For a half-open span [s, e):
    the start is shifted by every insertion at or before s (`pos <= s`), and
    the end by every insertion strictly before e (`pos < e`). The asymmetric
    rule keeps spans that lie inside, around, or across a bolded header from
    bleeding into the inserted <u><strong> or </strong></u> tag characters.
    """
    adjusted: list[tuple[int, int]] = []
    for s, e in spans:
        start_shift = sum(amount for pos, amount in shifts if pos <= s)
        end_shift = sum(amount for pos, amount in shifts if pos < e)
        adjusted.append((s + start_shift, e + end_shift))
    return adjusted


def load_predictions(
    inference_file: Path,
    icd_descriptions: dict[str, dict[str, str]],
) -> dict[str, list[dict]]:
    """Read the inference CSV into a {note_id: [prediction, ...]} map.

    Each prediction is sorted by descending probability inside its note, and
    the per-prediction evidence_data list of (span, attribution) tuples is
    sorted by descending attribution. Stable ordering keeps the rendered
    sidebar deterministic across reruns and saves the JS the work of finding
    a max-attribution fallback.
    """
    if not inference_file.exists():
        raise FileNotFoundError(f"Inference results file not found: {inference_file}")

    predictions: dict[str, list[dict]] = {}
    rows_seen = 0
    with inference_file.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing = [c for c in REQUIRED_INFERENCE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{inference_file} is missing required columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            rows_seen += 1
            note_id = row["note_id"]
            predicted_code = row["predicted_code"]
            probability = float(row["predicted_code_probability"])

            raw_spans = ast.literal_eval(row["evidence_spans"])
            spans = [tuple(span) for span in raw_spans] if isinstance(raw_spans, list) else []
            raw_attrs = ast.literal_eval(row["evidence_attributions"])
            attributions = [float(a) for a in raw_attrs] if isinstance(raw_attrs, list) else []

            evidence_data = sorted(
                zip(spans, attributions),
                key=lambda item: item[1],
                reverse=True,
            )

            normalized = normalize_icd_code(predicted_code)
            info = icd_descriptions.get(
                normalized,
                {"title": "Unknown code", "type": "unknown", "version": "unknown"},
            )

            predictions.setdefault(note_id, []).append({
                "predicted_code": predicted_code,
                "formatted_code": format_icd_code(normalized),
                "long_title": info["title"],
                "code_type": info["type"],
                "code_version": info["version"],
                "predicted_code_probability": probability,
                "evidence_data": evidence_data,
            })

    for note_id, items in predictions.items():
        items.sort(key=lambda item: (-item["predicted_code_probability"], item["predicted_code"]))

    LOGGER.info("Loaded %d predictions across %d notes", rows_seen, len(predictions))
    return predictions


def js_string_literal(text: str) -> str:
    """JSON-encode a Python string into a JS-safe string literal.

    Forces ASCII so that JSON line/paragraph separators (U+2028, U+2029),
    which are valid in JSON but illegal inside a JS string literal, are
    safely escaped. Also rewrites any literal ``</`` so the resulting HTML
    cannot be terminated by an embedded ``</script>``.
    """
    return json.dumps(text, ensure_ascii=True).replace("</", "<\\/")


def js_json_literal(value) -> str:
    """JSON-encode a Python value for embedding inside a <script> tag."""
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def render_html(note_id: str, bolded_text: str, evidence_spans_data: dict) -> str:
    evidence_json = js_json_literal(evidence_spans_data)
    original_text_js = js_string_literal(bolded_text)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document {note_id}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            color: #333;
            background-color: #fafafa;
        }}
        #main-container {{
            display: flex;
            height: 100vh;
        }}
        #main-text {{
            flex: 1;
            overflow-y: auto;
            padding: 0;
            background-color: #ffffff;
            display: flex;
            flex-direction: column;
        }}
        #main-text h1 {{
            position: sticky;
            top: 0;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            z-index: 1;
        }}
        #text-content {{
            padding: 20px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
            line-height: 1.6;
            flex: 1;
        }}
        #side-bar {{
            width: 300px;
            overflow-y: auto;
            border-left: 1px solid #e0e0e0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #fff59d;
        }}
        h2 {{
            font-size: 18px;
            color: #424242;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        .code-list {{
            list-style-type: none;
            padding: 0;
        }}
        .code-list li {{
            margin-bottom: 10px;
        }}
        .code-list label {{
            margin-left: 5px;
            cursor: pointer;
            font-size: 16px;
            color: #616161;
        }}
        input[type="radio"] {{
            cursor: pointer;
        }}
        strong {{
            font-weight: bold;
        }}
        #thresholds {{
            margin-bottom: 20px;
        }}
        #thresholds label {{
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #424242;
        }}
        #thresholds input[type="range"] {{
            width: 100%;
        }}
        .threshold-value {{
            font-weight: bold;
            color: #1565c0;
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <div id="main-text">
            <h1>Document {note_id}</h1>
            <div id="text-content"></div>
        </div>
        <div id="side-bar">
            <h2>Predicted Codes</h2>
                <div id="thresholds">
                    <label for="code-threshold">
                        Code Prediction Confidence (&ge; <span id="code-threshold-value" class="threshold-value"></span>)
                    </label>
                    <input type="range" id="code-threshold" min="0.4" max="1" step="0.01" value="0.7">

                    <label for="evidence-threshold">
                        Evidence Confidence (&ge; <span id="evidence-threshold-value" class="threshold-value"></span>)
                    </label>
                    <input type="range" id="evidence-threshold" min="0.0" max="0.02" step="0.001" value="0.01">
                </div>
            <form>
                <ul class="code-list" id="code-list">
                    <!-- Code items will be dynamically populated -->
                </ul>
            </form>
        </div>
    </div>
    <script>
        var evidence_spans_data = {evidence_json};
        var originalText = {original_text_js};

        var codeThresholdInput = document.getElementById('code-threshold');
        var codeThresholdValue = document.getElementById('code-threshold-value');
        var evidenceThresholdInput = document.getElementById('evidence-threshold');
        var evidenceThresholdValue = document.getElementById('evidence-threshold-value');
        var codeList = document.getElementById('code-list');
        var textContent = document.getElementById('text-content');
        var selectedCode = null;

        function updateThresholdValues() {{
            codeThresholdValue.textContent = parseFloat(codeThresholdInput.value).toFixed(2);
            evidenceThresholdValue.textContent = parseFloat(evidenceThresholdInput.value).toFixed(3);
        }}

        function computeFilteredSpans(data, evidenceThreshold) {{
            var filtered = data.evidence_spans.filter(function(item) {{
                return item.attribution >= evidenceThreshold;
            }});
            if (filtered.length === 0 && data.evidence_spans.length > 0) {{
                // Predictions are pre-sorted by descending attribution, so the
                // top-attribution span is always the first element.
                filtered = [data.evidence_spans[0]];
            }}
            return filtered;
        }}

        function filterCodes() {{
            var codeThreshold = parseFloat(codeThresholdInput.value);
            var evidenceThreshold = parseFloat(evidenceThresholdInput.value);
            var prevSelectedCode = selectedCode;

            codeList.innerHTML = '';

            Object.keys(evidence_spans_data).forEach(function(code) {{
                var data = evidence_spans_data[code];
                if (data.probability < codeThreshold) {{
                    return;
                }}
                var filteredSpans = computeFilteredSpans(data, evidenceThreshold);
                if (filteredSpans.length === 0) {{
                    return;
                }}
                data.filtered_spans = filteredSpans;

                var listItem = document.createElement('li');
                var radioInput = document.createElement('input');
                radioInput.type = 'radio';
                radioInput.name = 'code';
                radioInput.value = code;
                radioInput.id = 'code_' + code;
                if (code === prevSelectedCode) {{
                    radioInput.checked = true;
                }}
                radioInput.addEventListener('change', function() {{
                    if (this.checked) {{
                        onCodeSelected(this.value);
                    }}
                }});
                var label = document.createElement('label');
                label.htmlFor = 'code_' + code;
                var codeTypeLabel = data.code_type === 'diagnosis'
                    ? '[Dx]'
                    : (data.code_type === 'procedure' ? '[Px]' : '[?]');
                label.innerHTML = codeTypeLabel + ' <strong>' + data.formatted_code + '</strong>: ' + data.long_title;
                listItem.appendChild(radioInput);
                listItem.appendChild(label);
                codeList.appendChild(listItem);
            }});

            if (prevSelectedCode && evidence_spans_data[prevSelectedCode] && evidence_spans_data[prevSelectedCode].filtered_spans) {{
                selectedCode = prevSelectedCode;
                highlightSelectedCode();
            }} else {{
                selectedCode = null;
                textContent.innerHTML = originalText;
            }}
        }}

        function highlightText(spans) {{
            var result = [];
            var last_index = 0;
            spans.sort(function(a, b) {{ return a[0] - b[0]; }});
            for (var i = 0; i < spans.length; i++) {{
                var start = spans[i][0];
                var end = spans[i][1];
                if (start > last_index) {{
                    result.push(originalText.substring(last_index, start));
                }}
                result.push('<span class="highlight">' + originalText.substring(start, end) + '</span>');
                last_index = end;
            }}
            if (last_index < originalText.length) {{
                result.push(originalText.substring(last_index));
            }}
            textContent.innerHTML = result.join('');
        }}

        function onCodeSelected(code) {{
            selectedCode = code;
            highlightSelectedCode();
        }}

        function highlightSelectedCode() {{
            var data = evidence_spans_data[selectedCode];
            if (data && data.filtered_spans) {{
                var spans = data.filtered_spans.map(function(item) {{ return item.span; }});
                highlightText(spans);
            }} else {{
                textContent.innerHTML = originalText;
            }}
        }}

        codeThresholdInput.addEventListener('input', function() {{
            updateThresholdValues();
            filterCodes();
        }});

        evidenceThresholdInput.addEventListener('input', function() {{
            updateThresholdValues();
            var evidenceThreshold = parseFloat(evidenceThresholdInput.value);
            var codeThreshold = parseFloat(codeThresholdInput.value);
            Object.keys(evidence_spans_data).forEach(function(code) {{
                var data = evidence_spans_data[code];
                if (data.probability >= codeThreshold) {{
                    data.filtered_spans = computeFilteredSpans(data, evidenceThreshold);
                }}
            }});
            if (selectedCode) {{
                highlightSelectedCode();
            }}
        }});

        document.addEventListener('DOMContentLoaded', function() {{
            updateThresholdValues();
            textContent.innerHTML = originalText;
            filterCodes();
        }});
    </script>
</body>
</html>
"""


def render_note(
    note_id: str,
    formatted_text: str,
    note_predictions: list[dict],
    section_literals: set[str],
) -> str:
    bolded_text, shifts = bold_literals_in_text(formatted_text, section_literals)

    evidence_spans_data: dict[str, dict] = {}
    for item in note_predictions:
        adjusted_spans_attributions = [
            {"span": adjust_evidence_spans([span], shifts)[0], "attribution": attribution}
            for span, attribution in item["evidence_data"]
        ]
        evidence_spans_data[item["predicted_code"]] = {
            "probability": item["predicted_code_probability"],
            "formatted_code": item["formatted_code"],
            "long_title": item["long_title"],
            "code_type": item["code_type"],
            "code_version": item["code_version"],
            "evidence_spans": adjusted_spans_attributions,
        }

    return render_html(note_id, bolded_text, evidence_spans_data)


def visualise(
    inference_file: Path,
    formatted_dir: Path,
    output_dir: Path,
) -> None:
    if not SECTION_PATTERNS_FILE.exists():
        raise FileNotFoundError(f"Section patterns file not found: {SECTION_PATTERNS_FILE}")
    if not formatted_dir.exists():
        raise FileNotFoundError(
            f"Formatted-texts directory not found: {formatted_dir}. "
            "Run ner/extract_entities.py with --save-formatted-texts first."
        )

    section_literals = load_section_literals(SECTION_PATTERNS_FILE)
    icd_descriptions = load_icd_descriptions()
    predictions = load_predictions(inference_file, icd_descriptions)

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(formatted_dir.glob("formatted_*.txt"))
    if not txt_files:
        LOGGER.warning("No formatted_*.txt files found under %s", formatted_dir)
        return

    written = 0
    skipped_no_predictions: list[str] = []
    for txt_file in txt_files:
        note_id = txt_file.stem[len("formatted_"):]
        if note_id not in predictions:
            skipped_no_predictions.append(note_id)
            continue

        formatted_text = txt_file.read_text(encoding="utf-8")
        html = render_note(note_id, formatted_text, predictions[note_id], section_literals)

        out_path = output_dir / f"{note_id}.html"
        out_path.write_text(html, encoding="utf-8")
        LOGGER.info("Wrote %s", out_path)
        written += 1

    if skipped_no_predictions:
        LOGGER.info(
            "Skipped %d formatted note(s) with no matching predictions (e.g. %s)",
            len(skipped_no_predictions),
            ", ".join(skipped_no_predictions[:3]),
        )

    notes_without_text = sorted(set(predictions) - {p.stem[len("formatted_"):] for p in txt_files})
    if notes_without_text:
        LOGGER.warning(
            "%d predicted note(s) have no formatted_<note_id>.txt under %s (e.g. %s). "
            "Re-run extract_entities.py with --save-formatted-texts to include them.",
            len(notes_without_text),
            formatted_dir,
            ", ".join(notes_without_text[:3]),
        )

    LOGGER.info("Generated %d HTML file(s) in %s", written, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inference_file",
        type=Path,
        help="Path to the inference results CSV with evidence spans, as produced by "
             "external/plm_ca/infer_with_explanations.py.",
    )
    parser.add_argument(
        "--formatted-dir",
        "--formatted_dir",
        dest="formatted_dir",
        type=Path,
        default=DEFAULT_FORMATTED_DIR,
        help=f"Directory containing formatted_<note_id>.txt files "
             f"(default: {DEFAULT_FORMATTED_DIR}). Point at a per-run "
             "subdirectory if stale formatted texts from earlier runs would "
             "otherwise be picked up.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the per-note HTML files into "
             f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        visualise(
            inference_file=args.inference_file,
            formatted_dir=args.formatted_dir,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
