#script for clustering .npy embedding files produced by Perch 2.0

import os
import glob
import numpy as np
import pandas as pd
import hdbscan
import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import concurrent.futures
import multiprocessing   

# ==========================================
# --- 1. RUN CONFIGURATION SWITCHBOARD ---
# ==========================================
#Options: 'br_active', 'fr_active', 'br_all_channels', 'fr_all_channels'
TARGET_RUN = 'br_active' 

# --- 2. HYPERPARAMETER GRID ---
GRID_PARAMS = {
    'metrics': ['euclidean'],        
    'hdbscan_min_cluster_size': [2], 
    'hdbscan_min_samples': [1],      
    'hdbscan_epsilon': [0.0],        
    'hierarchical_thresholds': [0.7] 
}

# --- 3. HARDCODED DIRECTORY PATHS ---
RUN_CONFIGS = {
    'fr_active': {
        'out_dir': '[root]/bacpipe_results/clusters/fr_active/grid_mappings',
        'dirs': {
            'STM_h': '[path to STM_h_fr_active embeddings]',
            'STM_s_HTR': '[path to STM_s_HTR_fr_active embeddings]',
            'STM_s_LTR': '[path to STM_s_LTR_fr_active embeddings]',
            'PER': '[path to PER_fr_active embeddings]'
        }
    },
    'fr_all_channels': {
        'out_dir': '[root]/bacpipe_results/clusters/fr_all_channels/grid_mappings',
        'dirs': {
            'STM_h': '[path to STM_h_fr_all_channels embeddings]',
            'STM_s_HTR': '[path to STM_s_HTR_fr_all_channels embeddings]',
            'STM_s_LTR': '[path to STM_s_LTR_fr_all_channels embeddings]',
            'PER': '[path to PER_fr_all_channels embeddings]'
        }
    },
    'br_active': {
        'out_dir': '[root]/bacpipe_results/clusters/br_active/grid_mappings',
        'dirs': {
            'STM_h': '[path to STM_h_br_active embeddings]',
            'STM_s_HTR': '[path to STM_s_HTR_br_active embeddings]',
            'STM_s_LTR': '[path to STM_s_LTR_br_active embeddings]',
            'PER': '[path to PER_br_active embeddings]'
        }
    },
    'br_all_channels': {
        'out_dir': '[root]/bacpipe_results/clusters/br_all_channels/grid_mappings',
        'dirs': {
            'STM_h': '[path to STM_h_br_all_channels embeddings]',
            'STM_s_HTR': '[path to STM_s_HTR_br_all_channels embeddings]',
            'STM_s_LTR': '[path to STM_s_LTR_br_all_channels embeddings]',
            'PER': '[path to PER_br_all_channels embeddings]'
        }
    }
}

# ==========================================

def load_and_pool_embeddings(file_paths):
    """Loads .npy files, averages time frames if 2D, and L2-normalizes them."""
    embeddings = []
    valid_files = []
    
    for f in file_paths:
        try:
            emb = np.load(f)
            if emb.ndim == 2:
                emb = np.mean(emb, axis=0)
            embeddings.append(emb)
            valid_files.append(os.path.basename(f))
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not embeddings:
        return None, []
        
    X = np.vstack(embeddings)
    X_norm = normalize(X, norm='l2')
    return X_norm, valid_files

def run_clustering_grid(file_prefix, file_paths, output_dir):
    """Runs a grid search of clustering parameters and saves a consolidated map."""
    if not file_paths:
        return {"n_embeddings": 0}

    print(f"[{file_prefix}] Loading {len(file_paths)} embeddings...")
    X, filenames = load_and_pool_embeddings(file_paths)
    
    if X is None or len(X) < 2:
        return {"n_embeddings": len(file_paths)}

    mapping_data = {'recording': filenames}
    stats = {"n_embeddings": len(filenames)}

    print(f"[{file_prefix}] Running Hierarchical grid...")
    for metric, thresh in itertools.product(GRID_PARAMS['metrics'], GRID_PARAMS['hierarchical_thresholds']):
        try:
            link_method = 'ward' if metric == 'euclidean' else 'average'
            
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, metric=metric, linkage=link_method)
            labels = clusterer.fit_predict(X)
            
            metric_short = "euc" if metric == "euclidean" else "man"
            col_name = f"hier_{metric_short}_t{thresh}"
            
            mapping_data[col_name] = labels
            stats[col_name] = len(set(labels))
        except Exception as e:
            print(f"[{file_prefix}] Warning: Hierarchical failed for {col_name}: {e}")

    mapping_df = pd.DataFrame(mapping_data)
    map_path = os.path.join(output_dir, f"{file_prefix}_cluster_map.csv")
    mapping_df.to_csv(map_path, index=False)
    
    print(f"[{file_prefix}] Completed! Mappings saved.")

    return stats

#Parallel worker function
def execute_task(task):
    """Wrapper function to be executed by a parallel worker."""
    subset_name, group_val, site_files, map_out_dir = task
    group_name = f"{subset_name}_{group_val}"
    file_prefix = f"{group_name}_{TARGET_RUN}"
    
    # Run the workload
    results = run_clustering_grid(file_prefix, site_files, map_out_dir)
    return group_name, results

def update_summary(summary_dict, group_name, results):
    """Horizontally appends stats to the master dictionary."""
    if group_name not in summary_dict:
        summary_dict[group_name] = {"site": group_name}
    for key, value in results.items():
        summary_dict[group_name][key] = value

def queue_tasks(tasks_list, df, subset_name, group_by_col, map_out_dir):
    """Helper function to build a queue of tasks instead of executing them sequentially."""
    for group_val, group_data in df.groupby(group_by_col):
        site_files = group_data['filepath'].tolist()
        tasks_list.append((subset_name, group_val, site_files, map_out_dir))

def main():
    print(f"\n================ Processing Target: {TARGET_RUN.upper()} ================")
    
    config = RUN_CONFIGS[TARGET_RUN]
    dirs = config['dirs']
    map_out_dir = config['out_dir']
    
    os.makedirs(map_out_dir, exist_ok=True)

    registry = []
    for dataset, path in dirs.items():
        file_pattern = os.path.join(path, "**", "*.npy")
        files = glob.glob(file_pattern, recursive=True)
        print(f"Found {len(files)} embeddings in {dataset}")
        
        for f in files:
            basename = os.path.basename(f)
            if basename.startswith('fr_') or basename.startswith('br_'):
                basename = basename[3:]
                
            parts = basename.split('_')
            
            if len(parts) >= 2:
                site_raw = parts[0]
                date_raw = parts[1].split('.')[0] 
                site_clean = site_raw.replace('.150m', '').replace('.0m', '')
                
                if dataset == "PER":
                    if date_raw == '20190116': date_label = "Day1"
                    elif date_raw == '20190120': date_label = "Day2"
                    elif date_raw == '20190131': date_label = "Day3"
                    else: date_label = date_raw
                    site_date = f"{site_clean}_{date_label}"
                else:
                    site_date = f"{site_clean}_{date_raw}"
                
                registry.append({
                    "filepath": f, 
                    "dataset": dataset, 
                    "Site": site_clean, 
                    "Site_Date": site_date
                })

    df = pd.DataFrame(registry)
    if df.empty:
        print(f"No valid embeddings found for {TARGET_RUN}. Check your paths!")
        return

    # ==========================================
    # --- TASK QUEUE (Custom Groupings) ---
    # ==========================================
    tasks = []

    print("\n--- Queuing Datasets ---")
    if 'PER' in df['dataset'].values:
        queue_tasks(tasks, df[df['dataset'] == 'PER'], 'PER', 'Site_Date', map_out_dir)
        
    if 'STM_h' in df['dataset'].values:
        queue_tasks(tasks, df[df['dataset'] == 'STM_h'], 'STM_h', 'Site', map_out_dir)
        
    if 'STM_s_HTR' in df['dataset'].values:
        queue_tasks(tasks, df[df['dataset'] == 'STM_s_HTR'], 'STM_s_HTR', 'Site', map_out_dir)
        
    if 'STM_s_LTR' in df['dataset'].values:
        queue_tasks(tasks, df[df['dataset'] == 'STM_s_LTR'], 'STM_s_LTR', 'Site', map_out_dir)

    stm_s_all_df = df[df['dataset'].isin(['STM_s_HTR', 'STM_s_LTR'])]
    if not stm_s_all_df.empty:
        queue_tasks(tasks, stm_s_all_df, 'STM_s_All', 'Site', map_out_dir)

    stm_all_15s_df = df[df['dataset'].isin(['STM_h', 'STM_s_HTR'])]
    if not stm_all_15s_df.empty:
        queue_tasks(tasks, stm_all_15s_df, 'STM_All_15s', 'Site', map_out_dir)

    stm_all_df = df[df['dataset'].isin(['STM_h', 'STM_s_HTR', 'STM_s_LTR'])]
    if not stm_all_df.empty:
        queue_tasks(tasks, stm_all_df, 'STM_All', 'Site', map_out_dir)

    # ==========================================
    # --- PARALLEL EXECUTION ENGINE ---
    # ==========================================
    summary_dict = {}
    
    # Automatically detect cores assigned by Slurm (fallback to CPU count if running locally)
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    print(f"\nLaunching {len(tasks)} clustering tasks across {num_cores} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the process pool
        future_to_task = {executor.submit(execute_task, task): task for task in tasks}
        
        # As each task completes, grab the result and update the summary
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                group_name, results = future.result()
                update_summary(summary_dict, group_name, results)
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    # ==========================================
    # --- Save Master Summary ---
    # ==========================================
    if not summary_dict:
        print("\nNo data processed. Master summary will not be saved.")
        return

    summary_df = pd.DataFrame(list(summary_dict.values()))
    summary_df.fillna(0, inplace=True)
    
    data_cols = sorted([c for c in summary_df.columns if c != 'site'])
    summary_df = summary_df[['site'] + data_cols]
    
    summary_path = os.path.join(map_out_dir, f"clustering_master_grid_summary_{TARGET_RUN}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nFinished! Master Grid Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
