#Script for calculating SR estimated by Perch at different thresholds

import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE ALL FILE PATHS & VERSIONS
# ==========================================
base_dir = '[root]'

master_file = os.path.join(base_dir, 'clusters/Master_Summary_SR_with_Clusters.csv')
output_file = os.path.join(base_dir, 'clusters/Master_Summary_SR_with_Clusters_and_Perch.csv')

per_species_file = os.path.join(base_dir, 'PER_species_list_ebird_with_codes.csv')
stm_species_file = os.path.join(base_dir, 'STM_species_list_ebird_with_codes.csv')

# Define the two model versions to iterate over
perch_versions = [
    {
        'name': 'V1', 
        'path': os.path.join(base_dir, 'classifiers/perch/perch_1'), 
        'score_col': 'standard_sigmoid'
    },
    {
        'name': 'V2', 
        'path': os.path.join(base_dir, 'classifiers/perch/perch_2'), 
        'score_col': 'calibrated_probability'
    }
]

# ==========================================
# 2. LOAD & TAG REGIONAL LISTS
# ==========================================
print("Loading regional species lists...")

df_per_sp = pd.read_csv(per_species_file)
# Combine both ebird_code (V1) and Scientific Name (V2) into a single lookup set
per_species_codes = set(df_per_sp['ebird_code'].dropna()).union(
    set(df_per_sp['Scientific Name'].dropna().astype(str).str.lower())
)

df_stm_sp = pd.read_csv(stm_species_file)
stm_species_codes = set(df_stm_sp['ebird_code'].dropna()).union(
    set(df_stm_sp['Scientific Name'].dropna().astype(str).str.lower())
)

# Initialize master dataframe and a list to track all new column names across versions
print(f"Loading Master Summary: {master_file}")
master_df = pd.read_csv(master_file, keep_default_na=False)
all_generated_columns = []

# ==========================================
# 3. PROCESS EACH PERCH VERSION
# ==========================================
for version in perch_versions:
    v_name = version['name']
    v_path = version['path']
    score_col = version['score_col']
    
    print(f"\n--- Processing Perch {v_name} ---")
    
    # File paths for current version
    perch_stm_h_file     = os.path.join(v_path, 'raw_logits_perch_STM_h.csv')
    perch_stm_s_htr_file = os.path.join(v_path, 'raw_logits_perch_STM_s.csv')
    perch_stm_s_ltr_file = os.path.join(v_path, 'raw_logits_perch_STM_s_LTR.csv')
    perch_per_file       = os.path.join(v_path, 'raw_logits_perch_PER.csv')
    
    print(f"Loading raw logits for {v_name}...")
    df_stm_h = pd.read_csv(perch_stm_h_file)
    df_stm_h['Dataset'], df_stm_h['region'] = 'STM_h', 'STM'

    df_stm_s_htr = pd.read_csv(perch_stm_s_htr_file)
    df_stm_s_htr['Dataset'], df_stm_s_htr['region'] = 'STM_s_HTR', 'STM'

    # Load the new LTR dataset
    if os.path.exists(perch_stm_s_ltr_file):
        df_stm_s_ltr = pd.read_csv(perch_stm_s_ltr_file)
        df_stm_s_ltr['Dataset'], df_stm_s_ltr['region'] = 'STM_s_LTR', 'STM'
    else:
        print(f" -> WARNING: STM_s_LTR file not found in {v_name}. Creating empty dataframe.")
        df_stm_s_ltr = pd.DataFrame()

    df_per = pd.read_csv(perch_per_file)
    df_per['Dataset'], df_per['region'] = 'PER', 'PER'

    # Combine into master Perch dataframe for this version
    perch_df = pd.concat([df_stm_h, df_stm_s_htr, df_stm_s_ltr, df_per], ignore_index=True)
    perch_df[score_col] = pd.to_numeric(perch_df[score_col], errors='coerce')

    # Boolean mask to flag regional species
    is_stm = perch_df['region'] == 'STM'
    is_per = perch_df['region'] == 'PER'
    in_stm_list = perch_df['species'].isin(stm_species_codes)
    in_per_list = perch_df['species'].isin(per_species_codes)
    perch_df['is_regional'] = (is_stm & in_stm_list) | (is_per & in_per_list)

    # ==========================================
    # 3b. BUILD HIERARCHICAL GROUPINGS
    # ==========================================
    print("Extracting hierarchical grouping labels from recordings...")

    # Strip prefixes and allow hyphens so STM_s properly extracts Site, Date, and Time
    rec_clean = perch_df['recording'].astype(str).str.replace(r'^(?:fr_|br_)', '', regex=True)
    extracted = rec_clean.str.extract(r'([A-Za-z0-9\.\-]+)_(\d{8})_(\d{6})')

    perch_df['Site_Raw'] = extracted[0]
    perch_df['Date'] = extracted[1]
    perch_df['Time'] = extracted[2]

    # Clean Site names exactly as we did in the Master script
    def clean_site(s):
        if pd.isna(s): return s
        s = str(s).replace('.150m', '').replace('.0m', '')
        reps = {'BEXTRA2': 'BExtraT2', 'BEXTRA3': 'BExtraT3', 'BEXTRAT2': 'BExtraT2', 'BEXTRAT3': 'BExtraT3'}
        return reps.get(s.upper(), s)

    perch_df['Site'] = perch_df['Site_Raw'].apply(clean_site)
    perch_df['Site_Date'] = perch_df['Site'] + "_" + perch_df['Date']

    # Build time intervals dynamically
    time_objs = pd.to_datetime(perch_df['Time'], format='%H%M%S', errors='coerce')

    # Mask specifically for high-temporal resolution STM to keep raw filename times
    stm_htr_mask = perch_df['Dataset'].isin(['STM_h', 'STM_s_HTR'])

    # Explicitly build the 15-second interval
    time_15s_floored = time_objs.dt.floor('15s').dt.strftime('%H%M%S').fillna('UnknownTime')
    time_to_use_15s = perch_df['Time'].fillna('UnknownTime').where(stm_htr_mask, time_15s_floored)
    perch_df['Site_Date_15sec'] = perch_df['Site_Date'] + "_" + time_to_use_15s

    intervals_to_run = [1, 2, 5, 10, 15, 30]

    for mins in intervals_to_run:
        floored_time = time_objs.dt.floor(f'{mins}min').dt.strftime('%H%M%S').fillna('UnknownTime')
        
        if mins == 1:
            time_to_use = perch_df['Time'].fillna('UnknownTime').where(stm_htr_mask, floored_time)
            perch_df['Site_Date_1min'] = perch_df['Site_Date'] + "_" + time_to_use
        else:
            perch_df[f'Site_Date_{mins}min'] = perch_df['Site_Date'] + "_" + floored_time

    # Map the base dataset labels
    dataset_labels = {
        'PER': 'All_PER_Combined', 
        'STM_h': 'All_STM_h', 
        'STM_s_HTR': 'All_STM_s_HTR',
        'STM_s_LTR': 'All_STM_s_LTR'
    }
    perch_df['entire_dataset_label'] = perch_df['Dataset'].map(dataset_labels)

    # Duplicating rows for Combined Dataset Rollups
    df_stm_all_15s = perch_df[perch_df['Dataset'].isin(['STM_h', 'STM_s_HTR'])].copy()
    df_stm_all_15s['Dataset'] = 'STM_All_15s'
    df_stm_all_15s['entire_dataset_label'] = 'All_STM_15s_Combined'

    df_stm_s_all = perch_df[perch_df['Dataset'].isin(['STM_s_HTR', 'STM_s_LTR'])].copy()
    df_stm_s_all['Dataset'] = 'STM_s_All'
    df_stm_s_all['entire_dataset_label'] = 'All_STM_s_Combined'

    df_stm_all = perch_df[perch_df['region'] == 'STM'].copy()
    df_stm_all['Dataset'] = 'STM_All'
    df_stm_all['entire_dataset_label'] = 'All_STM_Combined'

    # Merge the duplicate datasets back into the main run queue
    perch_df = pd.concat([perch_df, df_stm_all_15s, df_stm_s_all, df_stm_all], ignore_index=True)

    # ==========================================
    # 4. CALCULATE UNIQUE SR AT EACH THRESHOLD
    # ==========================================
    print(f"Calculating unique Species Richness at defined thresholds for {v_name}...")

    levels_to_calc = {
        'entire_dataset': 'entire_dataset_label',
        'Site': 'Site',
        'Site_Date': 'Site_Date',
        'Date': 'Date',
        '15_Second_Interval': 'Site_Date_15sec'
    }
    for mins in intervals_to_run:
        levels_to_calc[f'{mins}_Minute_Interval'] = f'Site_Date_{mins}min'

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    summary_dfs = []

    # Iterate through every hierarchical scale
    for summary_level, group_col in levels_to_calc.items():
        valid_rows = perch_df.dropna(subset=[group_col])
        if valid_rows.empty: continue
            
        grouped_base = valid_rows.groupby(['Dataset', group_col])
        level_res = pd.DataFrame(index=grouped_base.groups.keys())
        level_res.index.names = ['Dataset', 'Grouping']
        
        # Evaluate each threshold using the appropriate score column for this version
        for t in thresholds:
            t_int = int(t * 10)
            col_all = f'perch{v_name}>.{t_int}_all'
            col_reg = f'perch{v_name}>.{t_int}_region'
            
            # Add to tracking list so we can fill NaNs globally later
            if col_all not in all_generated_columns:
                all_generated_columns.extend([col_all, col_reg])
                
            above_t = valid_rows[valid_rows[score_col] > t]
            
            # Calculate TRUE SR using .nunique() for ALL species
            counts_all = above_t.groupby(['Dataset', group_col])['species'].nunique()
            level_res[col_all] = counts_all
            
            # Calculate TRUE SR using .nunique() for REGIONAL species
            above_t_reg = above_t[above_t['is_regional']]
            counts_reg = above_t_reg.groupby(['Dataset', group_col])['species'].nunique()
            level_res[col_reg] = counts_reg

        # Finalize format for merging
        level_res = level_res.reset_index()
        level_res['Summary_Level'] = summary_level
        summary_dfs.append(level_res)

    # Combine all Perch summaries for THIS version into one long dataframe
    version_summary = pd.concat(summary_dfs, ignore_index=True)

    # ==========================================
    # 5. MERGE THIS VERSION'S DATA WITH MASTER
    # ==========================================
    print(f"Merging {v_name} data with Master Spreadsheet...")
    master_df = pd.merge(
        master_df, 
        version_summary, 
        on=['Dataset', 'Summary_Level', 'Grouping'], 
        how='left'
    )

# ==========================================
# 6. FINALIZE & SAVE
# ==========================================
# Fill NaNs with 0 across all newly generated columns (for recordings/sites where Perch found 0 species)
master_df[all_generated_columns] = master_df[all_generated_columns].fillna(0).astype(int)

# Save the updated master spreadsheet
master_df.to_csv(output_file, index=False)

print(f"\nSuccess! Final dataset with Perch V1 and V2 estimators saved to:\n{output_file}")
