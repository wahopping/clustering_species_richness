
import pandas as pd
import os
import glob
import gc
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE FILE PATHS & DIRECTORIES
# ==========================================
master_csv_path = '/scratch/hpc/36/hopping/bacpipe_results/clusters/Master_Summary_SR.csv' 
output_csv = '/storage/hpc/36/hopping/clustering/Master_Summary_SR_with_Clusters.csv'

cluster_dirs = {
    '/scratch/hpc/36/hopping/bacpipe_results/clusters/fr_not_noise/grid_mappings': '_fr_noise_removed',
    '/scratch/hpc/36/hopping/bacpipe_results/clusters/br_not_noise/grid_mappings': '_br_noise_removed',
    '/scratch/hpc/36/hopping/bacpipe_results/clusters/fr_all_channels/grid_mappings': '_fr_all_channels',
    '/scratch/hpc/36/hopping/bacpipe_results/clusters/br_all_channels/grid_mappings': '_br_all_channels'
}

print("Loading Master Summary...")
# keep_default_na=False prevents pandas from stripping "NA" strings into empty NaN values
df_master = pd.read_csv(master_csv_path, keep_default_na=False)

def clean_site(site_str):
    s = str(site_str).replace('.150m', '').replace('.0m', '')
    reps = {'BEXTRA2': 'BExtraT2', 'BEXTRA3': 'BExtraT3', 'BEXTRAT2': 'BExtraT2', 'BEXTRAT3': 'BExtraT3'}
    return reps.get(s.upper(), s)

# Expanded to cover all the new dataset outputs
dataset_labels = {
    'PER': 'All_PER_Combined', 
    'STM_h': 'All_STM_h', 
    'STM_s_HTR': 'All_STM_s_HTR',
    'STM_s_LTR': 'All_STM_s_LTR',
    'STM_s_All': 'All_STM_s_Combined',
    'STM_All_15s': 'All_STM_15s_Combined',
    'STM_All': 'All_STM_Combined'
}

intervals_to_run = [1, 2, 5, 10, 15, 30]

# Pre-define the levels we need to calculate
levels_to_calc = {
    'entire_dataset': 'entire_dataset_label',
    'Site': 'Site',
    'Date': 'Date',          
    'Site_Date': 'Site_Date',
    '15_Second_Interval': 'Site_Date_15sec'
}
for mins in intervals_to_run:
    levels_to_calc[f'{mins}_Minute_Interval'] = f'Site_Date_{mins}min'

# ==========================================
# 2. CHUNKED PROCESSING (DIRECTORY BY DIRECTORY)
# ==========================================
for c_dir, suffix in cluster_dirs.items():
    print(f"\n" + "="*50)
    print(f"PROCESSING DIRECTORY: {suffix}")
    print("="*50)
    
    cluster_files = glob.glob(os.path.join(c_dir, '*_cluster_map.csv'))
    print(f" -> Found {len(cluster_files)} files. Ingesting & Compressing...")
    
    all_melted = []
    
    for fpath in cluster_files:
        fname = os.path.basename(fpath)

        # Skip redundant files
        if 'STM_s_HTR_Site_Date' in fname or 'STM_s_LTR_Site_Date' in fname:
            continue 

        # Parse the new prefixes (order matters to avoid substring overlap!)
        if fname.startswith('PER_'): dataset = 'PER'
        elif fname.startswith('STM_All_15s_'): dataset = 'STM_All_15s'
        elif fname.startswith('STM_All_'): dataset = 'STM_All'
        elif fname.startswith('STM_s_All_'): dataset = 'STM_s_All'
        elif fname.startswith('STM_s_HTR_'): dataset = 'STM_s_HTR'
        elif fname.startswith('STM_s_LTR_'): dataset = 'STM_s_LTR'
        elif fname.startswith('STM_h_'): dataset = 'STM_h'
        else: continue 
        
        df_c = pd.read_csv(fpath, keep_default_na=False)
        if 'recording' not in df_c.columns:
            continue
            
        method_cols = [c for c in df_c.columns if c != 'recording']
        
        df_melt = df_c.melt(id_vars=['recording'], value_vars=method_cols, var_name='method', value_name='cluster_id')
        df_melt = df_melt[df_melt['cluster_id'] != -1].copy()
        
        if df_melt.empty:
            continue
            
        # FIX 1: Prevent double-suffixing if the column already contains it
        df_melt['method'] = df_melt['method'].astype(str)
        df_melt['method'] = df_melt['method'].apply(lambda x: x if x.endswith(suffix) else x + suffix)
        
        # FIX 2: Prepend the filename to the cluster ID to ensure global uniqueness across files
        clean_fname = fname.replace('.csv', '')
        df_melt['cluster_id'] = clean_fname + "_" + df_melt['method'] + "_c" + df_melt['cluster_id'].astype(str)
        df_melt['Dataset'] = dataset
        
        # Metadata Extraction
        extracted = df_melt['recording'].astype(str).str.extract(r'([A-Za-z0-9\.]+)_(\d{8})_(\d{6})')
        df_melt['Site'] = extracted[0].apply(clean_site)
        df_melt['Date'] = extracted[1]
        df_melt['Time'] = extracted[2]
        
        df_melt['Site_Date'] = df_melt['Site'] + "_" + df_melt['Date']
        
        # Added: Native 15-second interval from filename timestamp
        df_melt['Site_Date_15sec'] = df_melt['Site_Date'] + "_" + df_melt['Time'].fillna('UnknownTime')

        time_objs = pd.to_datetime(df_melt['Time'], format='%H%M%S', errors='coerce')
        for mins in intervals_to_run:
            floored_time = time_objs.dt.floor(f'{mins}min').dt.strftime('%H%M%S').fillna('UnknownTime')
            
            if mins == 1:
                df_melt['Site_Date_1min'] = df_melt['Site_Date'] + "_" + floored_time
            else:
                df_melt[f'Site_Date_{mins}min'] = df_melt['Site_Date'] + "_" + floored_time

        df_melt['entire_dataset_label'] = dataset_labels.get(dataset, 'Unknown')

        cols_to_keep = ['Dataset', 'entire_dataset_label', 'Site', 'Date', 'Site_Date', 'Site_Date_15sec'] + [f'Site_Date_{m}min' for m in intervals_to_run] + ['method', 'cluster_id']
        df_melt = df_melt[cols_to_keep].drop_duplicates()
        all_melted.append(df_melt)

    if not all_melted:
        print(f" -> WARNING: No valid data extracted from {suffix}. Skipping.")
        continue

    print(f" -> Calculating cluster metrics for {suffix}...")
    df_dir_clusters = pd.concat(all_melted, ignore_index=True)
    
    del all_melted
    gc.collect()

    summary_dfs = []
    for summary_level, group_col in levels_to_calc.items():
        valid_rows = df_dir_clusters.dropna(subset=[group_col])
        if valid_rows.empty: continue
        
        grouped = valid_rows.groupby(['Dataset', group_col, 'method'])['cluster_id'].nunique().reset_index()
        grouped.rename(columns={group_col: 'Grouping'}, inplace=True)
        grouped['Summary_Level'] = summary_level
        summary_dfs.append(grouped)

    del df_dir_clusters
    gc.collect()

    print(f" -> Pivoting and merging {suffix} into Master Dataframe...")
    df_cluster_counts = pd.concat(summary_dfs, ignore_index=True)
    del summary_dfs
    
    df_dir_pivot = df_cluster_counts.pivot_table(
        index=['Dataset', 'Summary_Level', 'Grouping'], 
        columns='method', 
        values='cluster_id', 
        fill_value=0
    ).reset_index()
    del df_cluster_counts

    # FIX 3: Cleanly drop existing overlapping columns before merge to avoid _x / _y duplicates
    cols_to_merge = df_dir_pivot.columns.difference(['Dataset', 'Summary_Level', 'Grouping'])
    cols_to_drop = [c for c in cols_to_merge if c in df_master.columns]
    if cols_to_drop:
        df_master.drop(columns=cols_to_drop, inplace=True)

    # Merge this single directory's new columns onto the rolling Master dataframe
    df_master = pd.merge(df_master, df_dir_pivot, on=['Dataset', 'Summary_Level', 'Grouping'], how='left')
    
    del df_dir_pivot
    gc.collect()
    print(f" -> Merge complete for {suffix}.")

# ==========================================
# 3. FINAL CLEANUP AND EXPORT
# ==========================================
print("\n" + "="*50)
print("ALL DIRECTORIES PROCESSED.")
print("="*50)

df_master = df_master.fillna(0)
df_master.to_csv(output_csv, index=False)
print(f"Success! Final chunk-merged dataset saved to: {output_csv}")
