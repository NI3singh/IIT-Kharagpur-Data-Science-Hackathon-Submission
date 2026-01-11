"""
main.py - The Pipeline Orchestrator (The Conductor)

This is the main execution script that coordinates all three agents:
1. Agent 1: Claim Decomposer (decomposer.py)
2. Agent 2 & 3: Evidence Retriever + Judge (reasoner.py)

Data Flow:
    test.csv → Backstories → [Agent 1] → Claims → [Agent 2] → Evidence → [Agent 3] → Verdict → results.csv
"""

import os
import sys
import pandas as pd
import requests
from typing import List, Dict, Optional, Tuple
from time import time, sleep
from datetime import datetime
import traceback

# Import configuration
from config import (
    TEST_CSV,
    OUTPUT_DIR,
    OUTPUT_CSV,
    PATHWAY_HOST,
    PATHWAY_PORT,
    validate_environment
)

# Import the three-agent system
from decomposer import decompose_backstory
from reasoner import (
    retrieve_evidence_for_claims,
    judge_consistency,
    get_evidence_statistics
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Processing settings
VERBOSE_PER_ROW = False  # Set to True to see detailed logs for each row
SAVE_PROGRESS_EVERY = 10  # Save intermediate results every N rows

# Retry settings for failed rows
MAX_ROW_RETRIES = 2
RETRY_DELAY = 5


# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def validate_environment_setup() -> None:
    """
    Comprehensive pre-flight checks before processing.
    
    Validates:
    - Environment variables
    - Required files exist
    - Pathway server is running
    - Output directory exists
    """
    
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT RAG PIPELINE - PRE-FLIGHT CHECKS")
    print("="*70)
    
    # Check 1: Environment variables
    print("\n📋 Check 1: Environment Configuration")
    try:
        validate_environment()
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        sys.exit(1)
    
    # Check 2: Test CSV exists
    print(f"\n📋 Check 2: Input Data")
    if not os.path.exists(TEST_CSV):
        print(f"❌ Test CSV not found at: {TEST_CSV}")
        print(f"   Expected location: {TEST_CSV}")
        sys.exit(1)
    print(f"✓ Test CSV found: {TEST_CSV}")
    
    # Check CSV structure
    try:
        df = pd.read_csv(TEST_CSV)
        print(f"✓ CSV loaded: {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['id', 'content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Check 3: Pathway server health
    print(f"\n📋 Check 3: Pathway Vector Store")
    if not check_pathway_health():
        print(f"❌ Pathway server not accessible at http://{PATHWAY_HOST}:{PATHWAY_PORT}")
        print(f"\n   Start the server with:")
        print(f"   python pathway_app/app.py")
        sys.exit(1)
    print(f"✓ Pathway server is running")
    
    # Check 4: Output directory
    print(f"\n📋 Check 4: Output Directory")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"✓ Created output directory: {OUTPUT_DIR}")
    else:
        print(f"✓ Output directory exists: {OUTPUT_DIR}")
    
    print("\n" + "="*70)
    print("✅ ALL PRE-FLIGHT CHECKS PASSED")
    print("="*70)


def check_pathway_health() -> bool:
    """
    Health check for Pathway vector store server.
    
    Returns:
        bool: True if server is accessible, False otherwise
    """
    try:
        # Try a minimal retrieve request
        response = requests.post(
            f"http://{PATHWAY_HOST}:{PATHWAY_PORT}/v1/retrieve",
            json={"query": "test", "k": 1},
            timeout=30
        )
        return True
    except:
        return False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data() -> pd.DataFrame:
    """
    Load and validate test data from CSV.
    
    Returns:
        pd.DataFrame: Loaded test data
    """
    
    print("\n" + "="*70)
    print("📂 LOADING TEST DATA")
    print("="*70)
    
    df = pd.read_csv(TEST_CSV)
    
    print(f"✓ Loaded {len(df)} test cases")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Show sample
    print(f"\n📊 Sample row:")
    print(df.iloc[0].to_dict())
    
    return df


def prepare_backstory_text(row: pd.Series) -> str:
    """
    Extract and prepare backstory text from a CSV row.
    
    Combines caption + content if both exist, otherwise uses content only.
    
    Args:
        row: Pandas Series representing a CSV row
        
    Returns:
        str: Prepared backstory text
    """
    
    # Get caption (if exists)
    caption = ""
    if 'caption' in row and pd.notna(row['caption']) and str(row['caption']).strip():
        caption = str(row['caption']).strip()
    
    # Get content (required)
    content = ""
    if 'content' in row and pd.notna(row['content']) and str(row['content']).strip():
        content = str(row['content']).strip()
    
    # Combine
    if caption and content:
        # Both exist - combine with clear separation
        backstory = f"{caption}\n\n{content}"
    elif content:
        # Only content
        backstory = content
    elif caption:
        # Only caption (rare)
        backstory = caption
    else:
        # Neither - empty
        backstory = ""
    
    return backstory


# ============================================================================
# CORE PROCESSING PIPELINE
# ============================================================================

def process_single_backstory(
    row_id: str,
    backstory: str,
    verbose: bool = False
) -> Dict:
    """
    Process a single backstory through the three-agent pipeline.
    
    Pipeline:
        Backstory → [Agent 1] → Claims → [Agent 2] → Evidence → [Agent 3] → Verdict
    
    Args:
        row_id: Unique identifier for this row
        backstory: The backstory text to analyze
        verbose: Print detailed progress
        
    Returns:
        Dict with keys: id, prediction, rationale, status, error (if failed)
    """
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"🔄 Processing: {row_id}")
        print(f"{'─'*70}")
    
    start_time = time()
    
    try:
        # Validation
        if not backstory or not backstory.strip():
            return {
                "id": row_id,
                "prediction": None,
                "rationale": "Empty backstory",
                "status": "error",
                "error": "Empty or missing backstory text"
            }
        
        # ====================================================================
        # AGENT 1: CLAIM DECOMPOSITION
        # ====================================================================
        
        if verbose:
            print(f"\n🎯 AGENT 1: Decomposing backstory...")
        
        claims = decompose_backstory(
            backstory=backstory,
            verbose=verbose
        )
        
        if not claims:
            return {
                "id": row_id,
                "prediction": None,
                "rationale": "No claims extracted",
                "status": "error",
                "error": "Decomposer returned no claims"
            }
        
        if verbose:
            print(f"✓ Extracted {len(claims)} claims")
        
        # ====================================================================
        # AGENT 2: EVIDENCE RETRIEVAL
        # ====================================================================
        
        if verbose:
            print(f"\n🔍 AGENT 2: Retrieving evidence...")
        
        evidence = retrieve_evidence_for_claims(
            claims=claims,
            verbose=verbose
        )
        
        if not evidence:
            return {
                "id": row_id,
                "prediction": None,
                "rationale": "No evidence found in novel",
                "status": "error",
                "error": "Evidence retrieval returned no results"
            }
        
        if verbose:
            print(f"✓ Retrieved {len(evidence)} evidence chunks")
        
        # ====================================================================
        # AGENT 3: CONSISTENCY JUDGMENT
        # ====================================================================
        
        if verbose:
            print(f"\n⚖️  AGENT 3: Judging consistency...")
        
        prediction, rationale = judge_consistency(
            backstory=backstory,
            evidence_list=evidence,
            verbose=verbose
        )
        
        # ====================================================================
        # SUCCESS
        # ====================================================================
        
        processing_time = time() - start_time
        
        if verbose:
            print(f"\n✅ Complete in {processing_time:.2f}s")
            print(f"   Verdict: {'CONSISTENT' if prediction == 1 else 'CONTRADICT'}")
        
        return {
            "id": row_id,
            "prediction": int(prediction),  # Ensure integer, not float
            "rationale": rationale,
            "status": "success",
            "processing_time": processing_time,
            "num_claims": len(claims),
            "num_evidence": len(evidence)
        }
    
    except Exception as e:
        # Catch any errors and log them
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        if verbose:
            print(f"\n❌ Error processing {row_id}: {error_msg}")
            print(f"   Traceback:\n{error_trace}")
        
        return {
            "id": row_id,
            "prediction": None,
            "rationale": f"Processing error: {error_msg}",
            "status": "error",
            "error": error_msg,
            "error_trace": error_trace
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_backstories(
    df: pd.DataFrame,
    verbose_per_row: bool = False
) -> List[Dict]:
    """
    Process all backstories in the dataset with progress tracking.
    
    Args:
        df: DataFrame with test data
        verbose_per_row: Show detailed logs for each row
        
    Returns:
        List of result dictionaries
    """
    
    total_rows = len(df)
    results = []
    successful = 0
    failed = 0
    
    print("\n" + "="*70)
    print(f"🔄 PROCESSING {total_rows} BACKSTORIES")
    print("="*70)
    
    start_time = time()
    
    for idx, row in df.iterrows():
        row_num = idx + 1
        
        # Extract data
        row_id = str(row.get('id', idx))
        backstory = prepare_backstory_text(row)
        
        # Progress indicator
        print(f"\n[{row_num}/{total_rows}] Processing ID: {row_id}")
        print(f"   Backstory length: {len(backstory)} characters")
        
        # Process with retry logic
        result = None
        for attempt in range(1, MAX_ROW_RETRIES + 1):
            try:
                result = process_single_backstory(
                    row_id=row_id,
                    backstory=backstory,
                    verbose=verbose_per_row
                )
                
                # Success
                if result['status'] == 'success':
                    successful += 1
                    print(f"   ✅ {'CONSISTENT' if result['prediction'] == 1 else 'CONTRADICT'}")
                else:
                    failed += 1
                    print(f"   ❌ Error: {result.get('error', 'Unknown')}")
                
                break  # Success, exit retry loop
            
            except Exception as e:
                if attempt < MAX_ROW_RETRIES:
                    print(f"   ⚠️  Attempt {attempt} failed: {e}")
                    print(f"   ⏳ Retrying in {RETRY_DELAY}s...")
                    sleep(RETRY_DELAY)
                else:
                    # All retries failed
                    print(f"   ❌ Failed after {MAX_ROW_RETRIES} attempts")
                    result = {
                        "id": row_id,
                        "prediction": None,
                        "rationale": f"Processing failed after {MAX_ROW_RETRIES} attempts",
                        "status": "error",
                        "error": str(e)
                    }
                    failed += 1
        
        results.append(result)
        
        # Progress summary
        elapsed = time() - start_time
        avg_time = elapsed / row_num
        remaining = (total_rows - row_num) * avg_time
        
        print(f"   ⏱️  Elapsed: {elapsed:.1f}s | Avg: {avg_time:.1f}s/row | ETA: {remaining/60:.1f}m")
        print(f"   📊 Progress: {successful} success, {failed} failed")
        
        # Save intermediate results
        if row_num % SAVE_PROGRESS_EVERY == 0:
            save_intermediate_results(results, row_num)
    
    # Final summary
    total_time = time() - start_time
    
    print("\n" + "="*70)
    print("✅ BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"   Total processed: {total_rows}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/total_rows*100:.1f}%")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average: {total_time/total_rows:.1f}s per backstory")
    print("="*70)
    
    return results


def save_intermediate_results(results: List[Dict], row_num: int) -> None:
    """Save intermediate results to avoid data loss."""
    
    intermediate_path = os.path.join(OUTPUT_DIR, f"intermediate_results_{row_num}.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(intermediate_path, index=False)
    
    print(f"\n   💾 Saved intermediate results to: {intermediate_path}")


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def export_results(results: List[Dict]) -> None:
    """
    Export results to CSV in competition format.
    
    Args:
        results: List of result dictionaries
    """
    
    print("\n" + "="*70)
    print("💾 EXPORTING RESULTS")
    print("="*70)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure proper column types
    df['id'] = df['id'].astype(str)
    
    # Convert prediction to integer (0 or 1), handling None/NaN
    df['prediction'] = df['prediction'].apply(
        lambda x: int(x) if pd.notna(x) and x in [0, 1] else -1
    )
    
    # Ensure rationale is string
    df['rationale'] = df['rationale'].fillna("").astype(str)

    # FIX: Create the renamed dataframe ONCE and use it for everything
    df_export = df.rename(columns={'id': 'Story ID'})
    
    # Select and order columns for submission
    output_columns = ['Story ID', 'prediction', 'rationale']
    
    # Add optional diagnostic columns (can be removed before submission)
    diagnostic_columns = ['status', 'processing_time', 'num_claims', 'num_evidence']
    available_diagnostic = [col for col in diagnostic_columns if col in df_export.columns]
    
    # Create submission CSV (minimal)
    submission_df = df_export[output_columns].copy()
    submission_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✓ Submission file saved: {OUTPUT_CSV}")
    print(f"   Rows: {len(submission_df)}")
    print(f"   Columns: {list(submission_df.columns)}")
    
    # Create diagnostic CSV (full details)
    # CORRECTED: Use df_export here to avoid KeyError
    diagnostic_path = os.path.join(OUTPUT_DIR, "results_with_diagnostics.csv")
    full_columns = output_columns + available_diagnostic
    
    # This line is now safe because df_export has 'Story ID'
    df_export[full_columns].to_csv(diagnostic_path, index=False)
    
    print(f"✓ Diagnostic file saved: {diagnostic_path}")
    
    # Summary statistics
    print(f"\n📊 Results Summary:")
    
    # Filter out failed rows (-1) for statistics
    predictions = df_export[df_export['prediction'] != -1]['prediction']
    
    if len(predictions) > 0:
        consistent_count = (predictions == 1).sum()
        contradict_count = (predictions == 0).sum()
        
        print(f"   Consistent (1): {consistent_count} ({consistent_count/len(predictions)*100:.1f}%)")
        print(f"   Contradict (0): {contradict_count} ({contradict_count/len(predictions)*100:.1f}%)")
    
    failed_count = (df_export['prediction'] == -1).sum()
    if failed_count > 0:
        print(f"   ⚠️  Failed: {failed_count}")
    
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    
    Steps:
    1. Validate environment
    2. Load test data
    3. Process all backstories
    4. Export results
    """
    
    print("\n" + "="*70)
    print("🎬 MULTI-AGENT RAG PIPELINE - MAIN EXECUTION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time()
    
    try:
        # Step 1: Pre-flight checks
        validate_environment_setup()
        
        # Step 2: Load data
        df = load_test_data()
        
        # Step 3: Process all backstories
        results = process_all_backstories(
            df=df,
            verbose_per_row=VERBOSE_PER_ROW
        )
        
        # Step 4: Export results
        export_results(results)
        
        # Final summary
        total_time = time() - pipeline_start
        
        print("\n" + "="*70)
        print("🎊 PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print(f"Results saved to: {OUTPUT_CSV}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        print("Partial results may have been saved to intermediate files.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Pipeline for Backstory Consistency Analysis"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs for each row"
    )
    parser.add_argument(
        "--test-single",
        type=str,
        help="Test with a single backstory ID from the CSV"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight checks only, don't process"
    )
    
    args = parser.parse_args()
    
    # Update verbose setting
    if args.verbose:
        VERBOSE_PER_ROW = True
    
    # Dry run mode
    if args.dry_run:
        validate_environment_setup()
        print("\n✅ Dry run complete. Environment is ready.")
        sys.exit(0)
    
    # Test single row mode
    if args.test_single:
        validate_environment_setup()
        df = load_test_data()
        
        # Find the row
        test_row = df[df['id'].astype(str) == args.test_single]
        
        if test_row.empty:
            print(f"❌ No row found with ID: {args.test_single}")
            sys.exit(1)
        
        row = test_row.iloc[0]
        backstory = prepare_backstory_text(row)
        
        print(f"\n🧪 Testing single backstory: {args.test_single}")
        result = process_single_backstory(
            row_id=args.test_single,
            backstory=backstory,
            verbose=True
        )
        
        print(f"\n📊 Result:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Rationale: {result['rationale']}")
        print(f"   Status: {result['status']}")
        
        sys.exit(0)
    
    # Normal execution
    main()