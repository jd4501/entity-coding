#!/usr/bin/env python3
"""
Entity-Coding Pipeline Runner

This script provides a convenient way to run the complete entity extraction
and ICD coding pipeline from a single command.

Usage:
    python run_pipeline.py input_notes.{parquet,csv} [output_prefix] [--max_workers N] [--visualize]

Example:
    python run_pipeline.py ner/sample_notes.parquet
    python run_pipeline.py data/notes.csv my_results --max_workers 2
    python run_pipeline.py ner/sample_notes.parquet --visualize  # Generate HTML visualizations of codes+evidence
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def print_header():
    """Print a header for the pipeline."""
    print("=" * 60)
    print("Entity-Coding Pipeline")
    print("   Named Entity Recognition → ICD Code Prediction → Interactive Visualization")
    print("=" * 60)


def validate_conda_env():
    """Check if we're in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'entitycoding':
        print("️   Warning: Not in 'entitycoding' conda environment")
        print("   Current environment:", conda_env or "None")
        print("   Run: conda activate entitycoding")
        return False
    print(f"Using conda environment: {conda_env}")
    return True


def validate_input_file(input_file):
    """Validate that the input file exists and has supported format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Check file format
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in ['.parquet', '.csv']:
        print(f"Error: Unsupported file format: {file_ext}")
        print("Supported formats: .parquet, .csv")
        return False
    
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    file_type = "parquet" if file_ext == '.parquet' else "CSV"
    print(f"Input {file_type} file found: {input_file} ({file_size:.1f} MB)")
    return True


def run_entity_extraction(input_file, entities_file, max_workers, save_formatted_texts=False):
    """Run the entity extraction step."""
    print("\n" + "─" * 40)
    print("Step 1: Extracting Named Entities")
    print("─" * 40)
    
    cmd = [
        sys.executable, "ner/extract_entities.py",
        input_file,
        "--output_file", entities_file,
        "--max_workers", str(max_workers),
        "--save_ner_docs"
    ]
    
    if save_formatted_texts:
        cmd.append("--save-formatted-texts")

    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        if not os.path.exists(entities_file):
            print(f"Entity extraction failed: Output file not created")
            return False
            
        print(f"Entity extraction completed in {time.time() - start_time:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Entity extraction failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_icd_coding(entities_file, output_prefix):
    """Run the ICD coding inference step."""
    print("\n" + "─" * 40)
    print("️  Step 2: ICD Code Prediction")
    print("─" * 40)
    
    # Change to PLM-CA directory for inference
    original_dir = os.getcwd()
    plm_ca_dir = "external/plm_ca"
    
    try:
        os.chdir(plm_ca_dir)
        
        # Adjust paths relative to PLM-CA directory
        rel_entities_file = os.path.join("../..", entities_file)
        rel_output_prefix = os.path.join("../..", output_prefix)
        
        cmd = [
            sys.executable, "infer_with_explanations.py",
            rel_entities_file, rel_output_prefix
        ]
        
        print(f"Running from {plm_ca_dir}: {' '.join(cmd[1:])}")  # Hide python path for clarity
        start_time = time.time()
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        # Check if output files were created
        csv_file = f"{output_prefix}.csv"
        parquet_file = f"{output_prefix}.parquet"
        
        os.chdir(original_dir)  # Return to original directory
        
        if not os.path.exists(csv_file) or not os.path.exists(parquet_file):
            print(f"ICD coding failed: Output files not created")
            return False
            
        print(f"ICD coding completed in {time.time() - start_time:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        os.chdir(original_dir)  # Make sure we return to original directory
        print(f"ICD coding failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        os.chdir(original_dir)  # Make sure we return to original directory
        print(f"ICD coding failed with error: {e}")
        return False


def run_visualization(entities_file, icd_results_file):
    """Run the visualization step."""
    print("\n" + "─" * 40)
    print("Step 3: Generating Interactive Visualizations")
    print("─" * 40)
    
    # Check if formatted texts directory exists
    if not os.path.exists('results/formatted_texts'):
        print("No formatted texts found. Visualization requires formatted text files.")
        print("Make sure entity extraction was run with --save-formatted-texts flag")
        return False
    
    # Check if ICD results file exists
    if not os.path.exists(icd_results_file):
        print(f"ICD results file not found: {icd_results_file}")
        return False
    
    cmd = [
        sys.executable, "code_evidence/visualise_predictions_explanations.py",
        icd_results_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        # Check if visualization files were created
        viz_dir = 'results/visualised_notes'
        if os.path.exists(viz_dir) and os.listdir(viz_dir):
            html_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
            print(f"Visualization completed in {time.time() - start_time:.1f}s")
            print(f"Generated {len(html_files)} HTML visualization files")
            return True
        else:
            print("No visualization files were generated")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def print_summary(input_file, entities_file, output_prefix, total_time, visualization_enabled=False):
    """Print a summary of the pipeline results."""
    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"️  Total processing time: {total_time:.1f} seconds")
    print(f"Input file: {input_file}")
    print(f"Entities file: {entities_file}")
    
    print(f"ICD coding results:")
    print(f"   • {output_prefix}.csv")
    print(f"   • {output_prefix}.parquet")
    
    # Try to show some stats
    try:
        import pandas as pd
        entities_df = pd.read_csv(entities_file)
        results_df = pd.read_csv(f"{output_prefix}.csv")
        
        print(f"\nProcessing Summary:")
        print(f"   • {entities_df['note_id'].nunique()} clinical notes processed")
        print(f"   • {len(entities_df)} entities extracted")
        
        print(f"   • {len(results_df)} ICD codes predicted")
        print(f"\nNext steps:")
        print(f"   • Examine results in {output_prefix}.csv")
        print(f"   • Each prediction includes evidence spans and attribution scores")
        
        if visualization_enabled:
            viz_dir = 'results/visualised_notes'
            if os.path.exists(viz_dir) and os.listdir(viz_dir):
                html_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
                print(f"\nInteractive Visualizations:")
                print(f"   • {len(html_files)} HTML files in {viz_dir}/")
                print(f"   • Open any .html file in your browser to explore predictions")
                print(f"   • Adjust thresholds to filter codes and evidence spans")
        
    except Exception as e:
        print(f"\n️  Could not load result statistics: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Entity-Coding pipeline: NER → ICD Coding → Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run_pipeline.py ner/sample_notes.parquet
        python run_pipeline.py data/notes.csv my_results --max_workers 2
        python run_pipeline.py ner/sample_notes.parquet --visualize  # With HTML visualizations
        python run_pipeline.py data/clinical_notes.csv results/batch_1 --max_workers 4 --visualize
            """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input file (.parquet or .csv, must contain 'note_id' and 'text' columns)"
    )
    parser.add_argument(
        "output_prefix",
        nargs='?',
        default=None,
        help="Output file prefix (default: derived from input filename)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for entity extraction (default: 2)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate interactive HTML visualizations of predictions and evidence"
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Validate environment and input
    if not validate_conda_env():
        print("\nTip: This pipeline requires the 'entitycoding' conda environment")
        sys.exit(1)
    
    if not validate_input_file(args.input_file):
        sys.exit(1)
    
    # Determine output filenames
    if args.output_prefix is None:
        # Default: input_file.parquet -> input_file_entities.csv and results/input_file_results
        input_stem = Path(args.input_file).stem
        entities_file = f"results/ner/{input_stem}_entities.csv"
        output_prefix = f"results/coded/{input_stem}_results"
    else:
        # Remove only the .csv or .parquet extension from the output prefix if present
        if args.output_prefix.endswith('.csv'):
            args.output_prefix = args.output_prefix[:-4]
        elif args.output_prefix.endswith('.parquet'):
            args.output_prefix = args.output_prefix[:-8]
        entities_file = f"{args.output_prefix}_entities.csv"
        output_prefix = f"{args.output_prefix}_icd_results"
    
    print(f"\nPipeline Configuration:")
    print(f"   • Input: {args.input_file}")
    print(f"   • Entities output: {entities_file}")
    print(f"   • ICD results: {output_prefix}.csv/.parquet")
    print(f"   • Mode: Standard ICD coding")
    print(f"   • Max workers: {args.max_workers}")
    if args.visualize:
        print(f"   • Visualization: Enabled (HTML files in results/visualised_notes/)")
    else:
        print(f"   • Visualization: Disabled (use --visualize to enable)")
    
    # Create output directories
    for filepath in [entities_file, f"{output_prefix}.csv"]:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
    
    # Run the pipeline
    start_time = time.time()
    
    # Step 1: Entity Extraction
    if not run_entity_extraction(args.input_file, entities_file, args.max_workers, args.visualize):
        print("\nPipeline failed at entity extraction step")
        sys.exit(1)
    
    # Step 2: ICD Coding
    if not run_icd_coding(entities_file, output_prefix):
        print("\nPipeline failed at ICD coding step")
        sys.exit(1)
    
    # Step 3: Visualization
    if args.visualize:
        icd_results_file = f"{output_prefix}.csv"
        if not run_visualization(entities_file, icd_results_file):
            print("\n️  Pipeline completed but visualization failed")
            print("    Main results are still available in CSV/Parquet files")
    
    # Success!
    total_time = time.time() - start_time
    print_summary(args.input_file, entities_file, output_prefix, total_time, args.visualize)


if __name__ == "__main__":
    main()
