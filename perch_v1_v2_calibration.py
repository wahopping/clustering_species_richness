#script to calibrate Perch v2 (so that it outputs easily interpretable scores) using Perch 1.0 outputs.
#Take the values this script produces and put them in perch_classification.py at the top 

import os
import glob
import numpy as np
import pandas as pd

# --- Configuration ---
PERCH_1_DIR = '[root]/classifiers/perch/perch_1'
PERCH_2_DIR = '[root]/classifiers/perch/perch_2'

THRESHOLDS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

def calibrated_sigmoid(x, offset, temperature):
    """Applies offset and temperature scaling to raw logits."""
    scaled_x = (x - offset) / temperature
    scaled_x = np.clip(scaled_x, -100, 100) # Prevent math overflow
    return 1 / (1 + np.exp(-scaled_x))

def main():
    print("=" * 65)
    print(" Perch Algebraic Logit Calibration Tool (Paired Files Only)")
    print("=" * 65)
    
    p1_files = glob.glob(os.path.join(PERCH_1_DIR, "*.csv"))
    p2_files = glob.glob(os.path.join(PERCH_2_DIR, "*.csv"))

    if not p1_files or not p2_files:
        print("\nERROR: Could not find CSV files in one or both directories.")
        return

    # Extract basenames to find matching files
    p1_dict = {os.path.basename(f): f for f in p1_files}
    p2_dict = {os.path.basename(f): f for f in p2_files}
    
    # Find the intersection of files present in BOTH directories
    common_filenames = set(p1_dict.keys()).intersection(set(p2_dict.keys()))
    
    if not common_filenames:
        print("\nERROR: No matching filenames found between the two directories.")
        return
        
    print(f"\nFound {len(common_filenames)} matching datasets to compare:")
    for name in common_filenames:
        print(f" - {name}")

    # 1. Load Perch 1.0 target probabilities (ONLY from common files)
    print("\nLoading paired Perch 1.0 probabilities...")
    p1_probs = []
    for filename in common_filenames:
        df = pd.read_csv(p1_dict[filename])
        col = 'standard_sigmoid' if 'standard_sigmoid' in df.columns else df.columns[-1]
        p1_probs.extend(df[col].dropna().values)
    p1_probs = np.array(p1_probs)

    # 2. Load Perch 2.0 raw logits (ONLY from common files)
    print("Loading paired Perch 2.0 raw logits...")
    p2_logits = []
    for filename in common_filenames:
        df = pd.read_csv(p2_dict[filename])
        col = 'raw_logit' if 'raw_logit' in df.columns else df.columns[-2]
        p2_logits.extend(df[col].dropna().values)
    p2_logits = np.array(p2_logits)

    print(f"\nData successfully loaded: {len(p1_probs):,} Perch 1.0 records, {len(p2_logits):,} Perch 2.0 records.")

    # 3. Calculate target counts
    target_counts = np.array([np.sum(p1_probs >= t) for t in THRESHOLDS])
    
    print("\nTarget detections from paired Perch 1.0 files:")
    for t, c in zip(THRESHOLDS, target_counts):
        print(f"  >= {t:<3} : {c:,} detections")

    # 4. ALGEBRAIC CALIBRATION
    print("\nSorting Perch 2.0 logits to find exact rank matches...")
    p2_logits_sorted = np.sort(p2_logits)[::-1]
    
    target_logits = []
    valid_thresholds = []
    valid_targets = []
    
    for t, c in zip(THRESHOLDS, target_counts):
        if c > 0:
            idx = min(c - 1, len(p2_logits_sorted) - 1)
            target_logits.append(p2_logits_sorted[idx])
            valid_thresholds.append(t)
            valid_targets.append(c)
            
    target_logits = np.array(target_logits)
    valid_thresholds = np.array(valid_thresholds)
    
    Y = np.log(valid_thresholds / (1 - valid_thresholds))
    best_temp, best_offset = np.polyfit(Y, target_logits, 1)

    print("\n" + "="*55)
    print("                 CALIBRATION COMPLETE")
    print("="*55)
    print(f"Optimal Offset:      {best_offset:.4f}")
    print(f"Optimal Temperature: {best_temp:.4f}")
    print("="*55)
    
    # 5. Display the final results
    final_probs = calibrated_sigmoid(p2_logits, best_offset, best_temp)
    final_counts = [np.sum(final_probs >= t) for t in THRESHOLDS]
    
    print("\nComparison at Optimal Parameters (Paired Data Only):")
    print(f"{'Threshold':<12} | {'Perch 1.0 (Target)':<20} | {'Perch 2.0 (Calibrated)'}")
    print("-" * 58)
    for t, target, final in zip(THRESHOLDS, target_counts, final_counts):
        print(f">= {t:<9} | {target:<20,} | {final:,}")

if __name__ == '__main__':
    main()
