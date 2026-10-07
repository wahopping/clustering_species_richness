import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================
# Master and Clusters
MASTER_CSV = "[path to input data]"
DATASETS_15S_CSV = "[path to input data]"

# New Acoustic Indices from HPC
INDICES_15S_CSV = "[root]/true_15s_metrics.csv"
INDICES_1MIN_CSV = "[root]/true_1min_metrics.csv"
INDICES_15MIN_CSV = "[root]/true_15min_metrics.csv"

# Output
OUTPUT_CSV = "[path to data.csv]"

# ==========================================
# 2. LOAD DATA & PURGE PHANTOM ROWS
# ==========================================
print("Loading datasets...")
master_df = pd.read_csv(MASTER_CSV, keep_default_na=False)
master_df.replace('NA', np.nan, inplace=True) 

# FIX: Force Grouping to uppercase to guarantee matches
master_df['Grouping'] = master_df['Grouping'].astype(str).str.upper().str.strip()

# FIX: Purge phantom zero rows. 
# Sorts by SR descending, so if there is a duplicate (e.g. SR=5 and SR=0), it keeps the 5.
master_df['SR'] = pd.to_numeric(master_df['SR'], errors='coerce').fillna(0)
master_df = master_df.sort_values('SR', ascending=False).drop_duplicates(
    subset=['Grouping', 'Summary_Level', 'Dataset'], 
    keep='first'
)

datasets_15s_df = pd.read_csv(DATASETS_15S_CSV)

# ==========================================
# 3. MERGE CLUSTER COLUMNS (br_3, fr_3)
# ==========================================
print("Merging br_3 and fr_3 for 15-second intervals...")
cluster_cols = ["br_3", "fr_3"]

# Clean the recording names to match Master format and FORCE uppercase
datasets_15s_df['recording_clean'] = datasets_15s_df['recording'].str.replace(r'\.150m|\.0m', '', regex=True, case=False)
datasets_15s_df['recording_clean'] = datasets_15s_df['recording_clean'].astype(str).str.upper().str.strip()

# Subset and drop duplicates to prevent row explosion
subset_15s = datasets_15s_df[['recording_clean'] + cluster_cols].drop_duplicates(subset=['recording_clean'])

# Merge into master
master_df = master_df.merge(
    subset_15s, 
    left_on='Grouping', 
    right_on='recording_clean', 
    how='left'
)

# Enforce the 15-second condition (null out clusters for non-15s rows)
non_15s_mask = master_df['Summary_Level'] != '15_Second_Interval'
master_df.loc[non_15s_mask, cluster_cols] = np.nan
master_df = master_df.drop(columns=['recording_clean'])

# ==========================================
# 4. MERGE ACOUSTIC INDICES (Base Temporal Scales)
# ==========================================
print("Merging true acoustic indices from HPC CSVs...")

df_15s = pd.read_csv(INDICES_15S_CSV)
df_1min = pd.read_csv(INDICES_1MIN_CSV)
df_15min = pd.read_csv(INDICES_15MIN_CSV)

# Stack them into one massive indices dataframe
all_indices = pd.concat([df_15s, df_1min, df_15min], ignore_index=True)
all_indices.replace('NA', np.nan, inplace=True)

# FIX: Force HPC Grouping to uppercase
all_indices['Grouping'] = all_indices['Grouping'].astype(str).str.upper().str.strip()

# Identify true metric columns (excluding clusters)
metric_cols = [c for c in all_indices.columns if (c.startswith('br_') or c.startswith('fr_')) and c not in cluster_cols]

all_indices = all_indices[['Grouping', 'Summary_Level'] + metric_cols]
all_indices = all_indices.drop_duplicates(subset=['Grouping', 'Summary_Level'])

# Merge into master using BOTH keys to ensure strict temporal scale matching
master_df = master_df.merge(
    all_indices,
    on=['Grouping', 'Summary_Level'],
    how='left'
)

# ==========================================
# 5. CALCULATE MACRO AVERAGES (Site and Site_Date)
# ==========================================
print("Calculating Macro averages (Site & Site_Date) for designated datasets...")

# Define which temporal scale builds the macro average for which dataset
macro_source_map = {
    'STM_s_LTR': '15_Minute_Interval',
    'STM_s_HTR': '15_Second_Interval',
    'STM_h': '1_Minute_Interval'
}

def extract_hierarchy(grouping_str):
    """Safely extracts Site and Site_Date from Grouping (Site_Date_Time)"""
    parts = str(grouping_str).rsplit('_', 2) # Splits from right to handle underscores in Site names
    if len(parts) == 3:
        site = parts[0]
        date = parts[1]
        return site, f"{site}_{date}"
    return None, None

for dataset, base_level in macro_source_map.items():
    print(f" -> Aggregating {dataset} using {base_level}...")
    
    # Isolate the base level rows for this dataset
    base_mask = (master_df['Dataset'] == dataset) & (master_df['Summary_Level'] == base_level)
    base_data = master_df.loc[base_mask].copy()
    
    if base_data.empty:
        print(f"    No base data found for {dataset}. Skipping.")
        continue
        
    # Ensure all metric columns are numeric so pandas can average them
    for col in metric_cols:
        base_data[col] = pd.to_numeric(base_data[col], errors='coerce')
        
    # Extract Site and Site_Date from the raw Grouping string
    base_data[['Calc_Site', 'Calc_Site_Date']] = base_data['Grouping'].apply(
        lambda x: pd.Series(extract_hierarchy(x))
    )
    
    # Calculate Mean by Site_Date
    site_date_means = base_data.groupby('Calc_Site_Date')[metric_cols].mean().reset_index()
    
    # Calculate Mean by Site
    site_means = base_data.groupby('Calc_Site')[metric_cols].mean().reset_index()
    
    # Inject Site_Date averages back into Master DF
    for _, row in site_date_means.iterrows():
        target_mask = (master_df['Dataset'] == dataset) & \
                      (master_df['Summary_Level'] == 'Site_Date') & \
                      (master_df['Grouping'] == row['Calc_Site_Date'])
                      
        if target_mask.any():
            master_df.loc[target_mask, metric_cols] = row[metric_cols].values
            
    # Inject Site averages back into Master DF
    for _, row in site_means.iterrows():
        target_mask = (master_df['Dataset'] == dataset) & \
                      (master_df['Summary_Level'] == 'Site') & \
                      (master_df['Grouping'] == row['Calc_Site'])
                      
        if target_mask.any():
            master_df.loc[target_mask, metric_cols] = row[metric_cols].values

# ==========================================
# 6. EXPORT
# ==========================================
master_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSuccess! Unified dataset with macro averages saved as:\n{OUTPUT_CSV}")
