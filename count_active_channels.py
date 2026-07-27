#script for counting the number of channels thought to be active (not noise) per 15s segment

import pandas as pd
import os
import re
from collections import Counter

# 1. Define the target number for your noise folders and columns
target_num = "6" # Change this as needed for future runs; this was used to test different iterations of the noise removal

csv_path = '[path to output spreadsheet for counts of active channels for each 15s file]

# Define the base directory and the specific dataset folder names
base_dir = '[path to 15s ground truth sound files]'
dataset_folders = ['STM_h', 'STM_s_HTR', 'PER']

def get_recording_counts(directory):
    """Counts matching .wav files, stripping prefixes, suffixes, and extensions."""
    counts = Counter()
    if not os.path.exists(directory):
        print(f"Warning: Directory not found - {directory}")
        return counts
        
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            # Strip the _sourceX.wav suffix (where \d represents any digit 0-9)
            base_name = re.sub(r'_source\d\.wav$', '', filename)
            
            # Strip the specific prefixes if they exist at the start of the string
            base_name = re.sub(r'^(fr_|br_)', '', base_name)
            
            # Increment the count for this clean base recording name
            counts[base_name] += 1
            
    return counts

# 2. Load the existing combined CSV
# KEEP_DEFAULT_NA=FALSE prevents pandas from converting the string "NA" into empty missing values
df = pd.read_csv(csv_path, keep_default_na=False)

# Initialize master counters to hold the tallies across all datasets
total_br_counts = Counter()
total_fr_counts = Counter()

# 3. Tally the files across all directories
print(f"Scanning directories for not_noise_{target_num}...")

for folder in dataset_folders:
    # Dynamically construct directory paths for the current dataset folder
    br_dir = f'{base_dir}/bird_range/{folder}/not_noise_{target_num}'
    fr_dir = f'{base_dir}/full_range/{folder}/not_noise_{target_num}'
    
    print(f"  -> Scanning {folder}...")
    
    # Add the counts from this folder to the master tallies
    total_br_counts += get_recording_counts(br_dir)
    total_fr_counts += get_recording_counts(fr_dir)

# 4. Map the master counts to new columns dynamically based on target_num
df[f'br_{target_num}'] = df['recording'].map(total_br_counts).fillna(0).astype(int)
df[f'fr_{target_num}'] = df['recording'].map(total_fr_counts).fillna(0).astype(int)

# 5. Save the updated DataFrame back to the exact same file
df.to_csv(csv_path, index=False)
print(f"Done! Added columns for {target_num} to existing CSV: {csv_path}")
