import os
import sys
import glob
import csv
import ast
import json
import re
import argparse

# Loads formatted notes after processing with `ner/extract_entities.py` with `save_formatted_texts = True`. 
# Also loads mapping files for ICD codes to their textual descriptions
# and loads code evidence from `modules/plm_ca/infer_with_explanations.py`
# Maps all of the data onto an interactive HTML projection of the note. 

def normalize_icd_code(code):
    """Normalize ICD code by removing '.' and converting to uppercase."""
    return code.replace('.', '').upper()

def format_icd_code(code):
    """Format ICD code by inserting '.' after the third character if needed."""
    if len(code) > 3:
        return code[:3] + '.' + code[3:]
    else:
        return code

def load_section_literals(filename):
    """Load section literals from the JSON file and normalize them."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize literals to lowercase to prevent duplicates
    literals = set(item['literal'].lower() for item in data.get('section_rules', []))
    return literals

def bold_literals_in_text(text, literals):
    """Find literals in text and wrap them with <strong> tags. Adjust indices accordingly."""
    # Build a list of all matches and their positions
    modifications = []
    for literal in literals:
        for match in re.finditer(re.escape(literal), text, re.IGNORECASE):
            start, end = match.start(), match.end()
            modifications.append((start, end))

    # Remove overlapping matches by keeping the longest match
    modifications = sorted(modifications, key=lambda x: (x[0], -(x[1]-x[0])))
    non_overlapping_mods = []
    last_end = -1
    for start, end in modifications:
        if start >= last_end:
            non_overlapping_mods.append((start, end))
            last_end = end

    # Adjust the text and keep track of index shifts
    new_text = []
    last_index = 0
    shifts = []  # List of tuples (pos, shift_amount)
    total_shift = 0
    for start, end in non_overlapping_mods:
        # Adjust start and end based on previous shifts
        adjusted_start = start + total_shift
        adjusted_end = end + total_shift
        # Append text before the match
        new_text.append(text[last_index:start])
        # Wrap the literal with <strong> tags
        bold_literal = f'<u><strong>{text[start:end]}</strong></u>'
        new_text.append(bold_literal)
        # Record the shift in indices
        shift_amount = len(bold_literal) - (end - start)
        shifts.append((start, shift_amount))
        total_shift += shift_amount
        # Update indices
        last_index = end
    # Append the remaining text
    new_text.append(text[last_index:])
    bolded_text = ''.join(new_text)
    return bolded_text, shifts

def adjust_evidence_spans(spans, shifts):
    """Adjust evidence spans indices based on the shifts."""
    adjusted_spans = []
    for span in spans:
        start, end = span
        total_shift = 0
        for shift_pos, shift_amount in shifts:
            if shift_pos <= start:
                total_shift += shift_amount
            elif shift_pos < end:
                total_shift += shift_amount
        adjusted_start = start + total_shift
        adjusted_end = end + total_shift
        adjusted_spans.append((adjusted_start, adjusted_end))
    return adjusted_spans

# Load section literals
section_literals = load_section_literals('ner/section_patterns.json')

# Load ICD code descriptions from both diagnosis and procedure files
icd_code_descriptions = {}

# Load diagnosis codes
diag_file = 'data/code_descriptions/d_icd_diagnoses.csv'
if os.path.exists(diag_file):
    with open(diag_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            icd_version = row['icd_version']
            if icd_version not in ['9', '10']:
                continue  # Skip if not ICD version 9 or 10
            icd_code = row['icd_code']
            long_title = row['long_title']
            
            # Normalize the ICD code
            normalized_code = normalize_icd_code(icd_code)
            # Store the mapping with code type prefix
            icd_code_descriptions[normalized_code] = {
                'title': long_title,
                'type': 'diagnosis',
                'version': icd_version
            }

# Load procedure codes
proc_file = 'data/code_descriptions/d_icd_procedures.csv'
if os.path.exists(proc_file):
    with open(proc_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            icd_version = row['icd_version']
            if icd_version not in ['9', '10']:
                continue  # Skip if not ICD version 9 or 10
            icd_code = row['icd_code']
            long_title = row['long_title']
            
            # Normalize the ICD code
            normalized_code = normalize_icd_code(icd_code)
            # Store the mapping with code type prefix
            icd_code_descriptions[normalized_code] = {
                'title': long_title,
                'type': 'procedure',
                'version': icd_version
            }

def find_inference_file(specified_file=None):
    """Find the inference results file, either from command line argument or default locations."""
    if specified_file:
        if os.path.exists(specified_file):
            return specified_file
        else:
            print(f" Error: Specified inference file not found: {specified_file}")
            sys.exit(1)
    
    # Try to find the inference results file in default locations
    inference_files = [
        'code_evidence/inferred_notes_with_evidence.csv',
        'results/sample_processing/inferred_notes_with_evidence_sample_notes_entities.csv',
        'results/inferred_notes_with_evidence.csv'
    ]
    
    for file_path in inference_files:
        if os.path.exists(file_path):
            return file_path
    
    print(" Error: Could not find inference results file. Expected one of:")
    for f in inference_files:
        print(f"   • {f}")
    print(" Or specify the file path as a command line argument:")
    print("   python visualise_predictions_explanations.py path/to/inference_results.csv")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML visualizations of ICD predictions with evidence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualise_predictions_explanations.py  # Use default file search
  python visualise_predictions_explanations.py results/my_results.csv  # Use specific file
        """
    )
    parser.add_argument(
        "inference_file",
        nargs='?',
        default=None,
        help="Path to the inference results CSV file with evidence spans"
    )
    
    args = parser.parse_args()
    
    # Find the inference file
    inference_file = find_inference_file(args.inference_file)
    
    # Read predictions
    prediction_results = {}  # note_id -> list of predicted codes with evidence spans

    print(f" Loading predictions from: {inference_file}")
    with open(inference_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            note_id = row['note_id']
            predicted_code = row['predicted_code']
            predicted_code_probability = float(row['predicted_code_probability'])
            evidence_spans = row['evidence_spans']
            evidence_attributions = row['evidence_attributions']

            # Parse evidence_spans safely using ast.literal_eval
            evidence_spans_list = ast.literal_eval(evidence_spans)
            # Ensure spans is a list of tuples
            if isinstance(evidence_spans_list, list):
                spans = [tuple(span) for span in evidence_spans_list]
            else:
                spans = []

            # Parse evidence_attributions
            evidence_attributions_list = ast.literal_eval(evidence_attributions)

            # Ensure evidence_attributions_list is a list of floats
            if isinstance(evidence_attributions_list, list):
                attributions = [float(attr) for attr in evidence_attributions_list]
            else:
                attributions = []

            # Zip spans and attributions
            evidence_data = list(zip(spans, attributions))

            # Normalize and format the predicted code
            normalized_code = normalize_icd_code(predicted_code)
            formatted_code = format_icd_code(normalized_code)
            # Get the long title and type from the ICD descriptions
            code_info = icd_code_descriptions.get(normalized_code, {'title': 'Unknown code', 'type': 'unknown', 'version': 'unknown'})
            long_title = code_info['title']
            code_type = code_info['type']
            code_version = code_info['version']

            # Store the data
            if note_id not in prediction_results:
                prediction_results[note_id] = []

            prediction_results[note_id].append({
                'predicted_code': predicted_code,
                'formatted_code': formatted_code,
                'long_title': long_title,
                'code_type': code_type,
                'code_version': code_version,
                'predicted_code_probability': predicted_code_probability,
                'evidence_data': evidence_data  # List of tuples (span, attribution)
            })

    # Process each formatted_{note_id}.txt file
    txt_files = glob.glob('results/formatted_texts/formatted_*.txt')

    for txt_file in txt_files:
        # Extract note_id from filename
        basename = os.path.basename(txt_file)
        if basename.startswith('formatted_') and basename.endswith('.txt'):
            note_id = basename[len('formatted_'):-len('.txt')]
            # Now process the file
            if note_id not in prediction_results:
                print(f"No predictions for note_id {note_id}")
                continue

            # Read the text
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Bold literals in text and get shifts
            bolded_text, shifts = bold_literals_in_text(text, section_literals)

            # Adjust evidence spans indices
            predicted_data = prediction_results[note_id]
            evidence_spans_data = {}
            for item in predicted_data:
                code = item['predicted_code']
                spans_attributions = item['evidence_data']
                # Adjust spans
                adjusted_spans_attributions = []
                for (span, attribution) in spans_attributions:
                    adjusted_span = adjust_evidence_spans([span], shifts)[0]
                    adjusted_spans_attributions.append({'span': adjusted_span, 'attribution': attribution})

                # Update the item
                item['adjusted_spans_attributions'] = adjusted_spans_attributions

                # Build evidence_spans_data
                evidence_spans_data[code] = {
                    'probability': item['predicted_code_probability'],
                    'formatted_code': item['formatted_code'],
                    'long_title': item['long_title'],
                    'code_type': item['code_type'],
                    'code_version': item['code_version'],
                    'evidence_spans': adjusted_spans_attributions  # List of dicts with 'span' and 'attribution'
                }

            # Convert evidence_spans_data to JSON
            evidence_spans_json = json.dumps(evidence_spans_data)

            # Generate the HTML content
            html_content = f'''<!DOCTYPE html>
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
        #threshold-values {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #757575;
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <div id="main-text">
            <h1>Document {note_id}</h1>
            <div id="text-content">{bolded_text}</div>
        </div>
        <div id="side-bar">
            <h2>Predicted Codes</h2>
                <div id="thresholds">
                    <label for="code-threshold">
                        Code Prediction Confidence
                        <span id="code-threshold-value" style="display: none;"></span>
                    </label>
                    <input type="range" id="code-threshold" min="0.4" max="1" step="0.01" value="0.7">

                    <label for="evidence-threshold">
                        Evidence Confidence (Attribution Threshold)
                        <span id="evidence-threshold-value" style="display: none;"></span>
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
        var evidence_spans_data = {evidence_spans_json};

        var originalText = `{bolded_text}`;

        var codeThresholdInput = document.getElementById('code-threshold');
        var codeThresholdValue = document.getElementById('code-threshold-value');
        var evidenceThresholdInput = document.getElementById('evidence-threshold');
        var evidenceThresholdValue = document.getElementById('evidence-threshold-value');
        var codeList = document.getElementById('code-list');
        var selectedCode = null;

        function updateThresholdValues() {{
            codeThresholdValue.textContent = codeThresholdInput.value;
            evidenceThresholdValue.textContent = evidenceThresholdInput.value;
        }}

        function filterCodes() {{
            var codeThreshold = parseFloat(codeThresholdInput.value);
            var evidenceThreshold = parseFloat(evidenceThresholdInput.value);

            // Keep track of the previously selected code
            var prevSelectedCode = selectedCode;

            // Clear code list
            codeList.innerHTML = '';

            var codes = Object.keys(evidence_spans_data);
            codes.forEach(function(code) {{
                var data = evidence_spans_data[code];
                if (data.probability >= codeThreshold) {{
                    // Filter evidence spans
                    var filteredSpans = data.evidence_spans.filter(function(item) {{
                        return item.attribution >= evidenceThreshold;
                    }});
                    if (filteredSpans.length === 0 && data.evidence_spans.length > 0) {{
                        // Include the highest attribution evidence span
                        filteredSpans = [data.evidence_spans.reduce(function(prev, current) {{
                            return (prev.attribution > current.attribution) ? prev : current
                        }})];
                    }}
                    if (filteredSpans.length > 0) {{
                        // Store filtered spans in data
                        data.filtered_spans = filteredSpans;
                        // Add code to list
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
                        var codeTypeIcon = data.code_type === 'diagnosis' ? '' : (data.code_type === 'procedure' ? '️' : '');
                        label.innerHTML = codeTypeIcon + ' <strong>' + data.formatted_code + '</strong>: ' + data.long_title;
                        listItem.appendChild(radioInput);
                        listItem.appendChild(label);
                        codeList.appendChild(listItem);
                    }}
                }}
            }});

            // If the previously selected code is still available, re-select it
            if (prevSelectedCode && evidence_spans_data[prevSelectedCode] && evidence_spans_data[prevSelectedCode].filtered_spans) {{
                selectedCode = prevSelectedCode;
                highlightSelectedCode();
            }} else {{
                // Reset text without highlights
                selectedCode = null;
                document.getElementById('text-content').innerHTML = originalText;
            }}
        }}

        function highlightText(spans) {{
            var text = originalText;
            var result = [];
            var last_index = 0;
            spans.sort(function(a, b) {{ return a[0] - b[0]; }});
            for (var i = 0; i < spans.length; i++) {{
                var start = spans[i][0];
                var end = spans[i][1];
                if (start > last_index) {{
                    result.push(text.substring(last_index, start));
                }}
                result.push('<span class="highlight">' + text.substring(start, end) + '</span>');
                last_index = end;
            }}
            if (last_index < text.length) {{
                result.push(text.substring(last_index));
            }}
            document.getElementById('text-content').innerHTML = result.join('');
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
                // Reset text without highlights
                document.getElementById('text-content').innerHTML = originalText;
            }}
        }}

        codeThresholdInput.addEventListener('input', function() {{
            updateThresholdValues();
            filterCodes();
        }});

        evidenceThresholdInput.addEventListener('input', function() {{
            updateThresholdValues();
            // Only need to update the evidence spans for the selected code
            var evidenceThreshold = parseFloat(evidenceThresholdInput.value);
            var codes = Object.keys(evidence_spans_data);

            codes.forEach(function(code) {{
                var data = evidence_spans_data[code];
                if (data.probability >= parseFloat(codeThresholdInput.value)) {{
                    var filteredSpans = data.evidence_spans.filter(function(item) {{
                        return item.attribution >= evidenceThreshold;
                    }});
                    if (filteredSpans.length === 0 && data.evidence_spans.length > 0) {{
                        // Include the highest attribution evidence span
                        filteredSpans = [data.evidence_spans.reduce(function(prev, current) {{
                            return (prev.attribution > current.attribution) ? prev : current
                        }})];
                    }}
                    data.filtered_spans = filteredSpans;
                }}
            }});

            // Update highlights if a code is selected
            if (selectedCode) {{
                highlightSelectedCode();
            }}
        }});

        document.addEventListener('DOMContentLoaded', function() {{
            updateThresholdValues();
            filterCodes();
            // Initialize with original text
            document.getElementById('text-content').innerHTML = originalText;
        }});
    </script>
</body>
</html>
'''

            # Create output directory and save the HTML file
            output_dir = 'results/visualised_notes'
            os.makedirs(output_dir, exist_ok=True)
            
            html_filename = f'{note_id}.html'
            html_filepath = os.path.join(output_dir, html_filename)
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"Generated HTML file for note_id {note_id}: {html_filepath}")

if __name__ == "__main__":
    main()
