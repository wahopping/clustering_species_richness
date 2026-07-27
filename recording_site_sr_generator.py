#script for tabulating species richness from all recordings, sites, datasets etc using the original annotations files

import pandas as pd 
import os 
from datetime import datetime, timedelta 
import warnings
warnings.filterwarnings('ignore')

# ========================================== 
# 1. DEFINE FILE PATHS 
# ========================================== 
per_input = "[path to PER annotations]"
stms_htr_input = "[path to STM_s_HTR annotations]" 
stms_ltr_input = "[path to STM_s_LTR annotations]"
stmh_input = "[path to STM_h annotations]"
output_dir = "[path to output folder]"
output_csv = os.path.join(output_dir, 'Master_Summary_SR.csv') 
stms_master_list = "[path to STM_s master list of files]"

all_summaries = [] 

# ========================================== 
# 2. PROCESS PER DATASET 
# ========================================== 
print("=== PROCESSING PER ===") 
try: 
    df_per = pd.read_csv(per_input, keep_default_na=False) 
    date_mapping = {'A': '20190116', 'B': '20190120', 'C': '20190131'} 
    file_str = df_per['Begin File'].astype(str).str.upper().str.replace('.WAV', '') 
    
    df_per['Site'] = "Site" + file_str.str[:-1] 
    df_per['Date'] = file_str.str[-1].map(date_mapping).fillna("UnknownDate") 
    df_per['Site_Date'] = df_per['Site'] + "_" + df_per['Date'] 
    df_per['entire_dataset'] = 'All_PER_Combined' 

    base_time = pd.to_datetime('05:00:00', format='%H:%M:%S') 
    intervals_min = [1, 2, 5, 10, 15, 30] 

    for mins in intervals_min: 
        bin_seconds = mins * 60 
        bin_start_sec = (df_per['Begin Time (s)'] // bin_seconds) * bin_seconds 
        bin_time = base_time + pd.to_timedelta(bin_start_sec, unit='s') 
        time_str = bin_time.dt.strftime('%H%M%S') 
        df_per[f'Site_Date_{mins}min'] = df_per['Site_Date'] + "_" + time_str 

    bin_start_sec_15s = (df_per['Begin Time (s)'] // 15) * 15 
    bin_time_15s = base_time + pd.to_timedelta(bin_start_sec_15s, unit='s') 
    df_per['Site_Date_15sec'] = df_per['Site_Date'] + "_" + bin_time_15s.dt.strftime('%H%M%S') 

    valid_per = df_per.dropna(subset=['species']).copy() 
    valid_per['species'] = valid_per['species'].astype(str).str.upper().str.strip() 

    valid_per['species'] = valid_per['species'].replace({ 
        'INWO/ELWO': 'INWO', 
        'THRAUPIDAE SP': 'THRAUPIDAE SP', 
        'THRAUPIDAE SP.': 'THRAUPIDAE SP', 
        'THRAUPIDE SP.': 'THRAUPIDAE SP', 
        'YMFL': 'YMFL?', 
        'WOODCREEPER SP.': 'WOODCREEPER SP.?', 
        'YMFL OR SOMETHING': 'YMFL OR SOMETHING?'  
    }) 

    is_bg = valid_per['background'].isin([1, '1', 1.0, '1.0']) 
    is_unknown = valid_per['species'].str.contains(r'\?', regex=True) 

    def get_filtered_species_set(species_series): 
        s = set(species_series.dropna()) 
        def has_any(taxa_list): return not s.isdisjoint(taxa_list) 
        
        if 'DOSP' in s and has_any(['GFDO', 'WTDO', 'RUQD']): s.discard('DOSP') 
        if 'CRYPTURELLUS SP.' in s and has_any(['BATI', 'LITI', 'CITI']): s.discard('CRYPTURELLUS SP.') 
        if 'GFDO/WTDO' in s and has_any(['GFDO', 'WTDO']): s.discard('GFDO/WTDO') 
        if 'LOSP' in s and has_any(['RBMA', 'MEPA', 'RAGM', 'CFMA']): s.discard('LOSP') 
        if 'PASP' in s and has_any(['RBMA', 'MEPA', 'RAGM', 'CFMA']): s.discard('PASP') 
        if 'PISP' in s and has_any(['RUPI', 'PVPI', 'SCPI', 'PLPI']): s.discard('PISP') 
        if 'SBCA/BBWR' in s and has_any(['SBCA', 'BBWR']): s.discard('SBCA/BBWR') 
        if 'TISP' in s and has_any(['CITI', 'BATI', 'GRTI', 'LITI']): s.discard('TISP') 
        if 'THSP' in s and has_any(['HATH', 'WNTH', 'BBTH']): s.discard('THSP') 
        if 'TRSP' in s and has_any(['BTTR', 'AMTR', 'GBTR', 'BCTR']): s.discard('TRSP') 
        if 'WRSP' in s and has_any(['BBWR']): s.discard('WRSP') 
        
        s.discard('MXFM')  
        s.discard('THRAUPIDAE SP')  
        return s 

    def apply_per_rules(species_series): return len(get_filtered_species_set(species_series)) 

    def calculate_per_sr(group_col, level_name): 
        if valid_per.empty: return None 
        valid_per[group_col] = valid_per[group_col].fillna(f'Unknown_{group_col}') 
        
        sr = valid_per[~is_unknown & ~is_bg].groupby(group_col)['species'].apply(apply_per_rules) 
        sr_wb = valid_per[~is_unknown].groupby(group_col)['species'].apply(apply_per_rules) 
        sr_wu = valid_per[~is_bg].groupby(group_col)['species'].apply(apply_per_rules) 
        sr_wb_wu = valid_per.groupby(group_col)['species'].apply(apply_per_rules) 
        
        res = pd.DataFrame({ 
            'SR': sr, 'SR_with_background': sr_wb,  
            'SR_with_unknowns': sr_wu, 'SR_with_background_and_unknowns': sr_wb_wu 
        }).reset_index()  
        
        if 'min' in group_col or 'sec' in group_col:
            freq_str = group_col.split('_')[-1].replace('min', 'min').replace('15sec', '15s')
            full_times = pd.date_range(start='05:00:00', end='05:59:59', freq=freq_str).strftime('%H%M%S')
            unique_site_dates = df_per['Site_Date'].unique()
            master_grid = [f"{sd}_{t}" for sd in unique_site_dates for t in full_times]
            master_df = pd.DataFrame({group_col: master_grid})
            res = pd.merge(master_df, res, on=group_col, how='left')
        
        res = res.fillna(0)
        for col in ['SR', 'SR_with_background', 'SR_with_unknowns', 'SR_with_background_and_unknowns']:
            res[col] = res[col].astype(int)
            
        res.rename(columns={group_col: 'Grouping'}, inplace=True) 
        res['Summary_Level'] = level_name 
        res['Dataset'] = 'PER' 
        print(f"  -> Generated {len(res)} rows for PER [{level_name}]") 
        return res

    all_summaries.append(calculate_per_sr('entire_dataset', 'entire_dataset')) 
    all_summaries.append(calculate_per_sr('Site', 'Site')) 
    all_summaries.append(calculate_per_sr('Site_Date', 'Site_Date')) 
    all_summaries.append(calculate_per_sr('Date', 'Date')) 
    for mins in intervals_min: 
        all_summaries.append(calculate_per_sr(f'Site_Date_{mins}min', f'{mins}_Minute_Interval')) 
    all_summaries.append(calculate_per_sr('Site_Date_15sec', '15_Second_Interval')) 

except Exception as e: 
    print(f"  -> ERROR processing PER dataset: {e}") 


# ========================================== 
# 3. PROCESS STM DATASETS 
# ========================================== 
print("\n=== PROCESSING STM ===") 

def get_filtered_stm_species_set(species_series, include_unknowns=False): 
    cleaned_series = species_series.dropna().astype(str).str.upper().str.replace('_', ' ').str.replace('  ', ' ').str.strip()
    s = set(cleaned_series) 
    
    invalid_labels = {
        'NO LABELS', 'NOTHING', 'UNIDENTIFIED', 
        'NONBIRD BROWNCAPUCHIN', 'NONBIRD MONKEY', 'NONE NONE'
    }
    
    if include_unknowns: 
        invalid_labels.discard('UNIDENTIFIED')
        
    s -= invalid_labels
        
    def apply_sp_filter(base_name, prefixes):
        nonlocal s 
        if isinstance(prefixes, str): prefixes = (prefixes,)
        variations = {f"{base_name} SP", f"{base_name} SP."}
        if s.isdisjoint(variations): return
        for valid_sp in s:
            if valid_sp not in variations and valid_sp.startswith(prefixes):
                s -= variations
                break
                
    # Genus and Family Level Filters
    apply_sp_filter('AMAZONA', 'AMAZONA')
    apply_sp_filter('ARA', 'ARA')
    apply_sp_filter('HYLOCHARIS/CHLORESTES', ('HYLOCHARIS', 'CHLORESTES'))
    apply_sp_filter('LORIOTUS', 'LORIOTUS')
    apply_sp_filter('PATAGIOENAS', 'PATAGIOENAS')
    apply_sp_filter('PHAETHORNIS', 'PHAETHORNIS')
    apply_sp_filter('PICIDAE', ('MELANERPES', 'DRYOBATES', 'CAMPEPHILUS', 'DYROCOPUS', 'CELEUS', 'PICULUS', 'COLAPTES'))
    apply_sp_filter('PSAROCOLIUS', 'PSAROCOLIUS')
    apply_sp_filter('PSITTACIDAE', ('BROTOGERIS', 'PYRRHUA', 'ARATINGA', 'GUARUBA', 'PSITTACARA'))
    apply_sp_filter('THRAUPIS', 'THRAUPIS')
    apply_sp_filter('THRAUPIDAE', ('CHLOROPHANES', 'CYANERPES', 'DACNIS', 'TANGARA', 'STILPNIA', 'IXOTHRAUPIS', 'THRAUPIS', 'RAMPHOCELUS', 'TACHYPHONUS', 'LORIOTUS'))
    
    hummingbird_genera = (
        'CAMPYLOPTERUS', 'GLAUCIS', 'HYLOCHARIS', 'CHLORESTES', 'PHAETHORNIS', 
        'THALURANIA', 'AMAZILIA', 'POLYTMUS', 'CHLOROSTILBON', 'FLORISUGA', 
        'HELIOTHRYX', 'TOPAZA', 'HELIODOXA', 'THRENETES', 'ANTHRACOTHORAX', 
        'LOPHORNIS', 'CHRYSURONIA', 'MICROCHERA', 'EUPETOMENA', 'EUTOXERES', 
        'ANDRODON', 'DORYFERA'
    )
    apply_sp_filter('TROCHILIDAE', hummingbird_genera)

    strix_variations = {"STRIX", "STRIX SP", "STRIX SP."}
    if not s.isdisjoint(strix_variations):
        for valid_sp in s:
            if valid_sp not in strix_variations and valid_sp.startswith('STRIX'):
                s -= strix_variations
                break
                
    return s 

def calculate_stm_sr(df, group_col, level_name, dataset_name): 
    if df.empty: return None 
    df[group_col] = df[group_col].fillna(f'Unknown_{group_col}') 
    
    sr = df.groupby(group_col)['Species'].apply(lambda x: len(get_filtered_stm_species_set(x, include_unknowns=False))) 
    sr_with_unknowns = df.groupby(group_col)['Species'].apply(lambda x: len(get_filtered_stm_species_set(x, include_unknowns=True))) 
    
    sr_df = pd.DataFrame({ 
        'SR': sr, 'SR_with_unknowns': sr_with_unknowns 
    }).fillna(0).astype(int).reset_index() 
    
    sr_df.rename(columns={group_col: 'Grouping'}, inplace=True) 
    sr_df['Dataset'] = dataset_name 
    sr_df['Summary_Level'] = level_name 
    sr_df['SR_with_background'] = sr_df['SR'] 
    sr_df['SR_with_background_and_unknowns'] = sr_df['SR_with_unknowns'] 
    print(f"  -> Generated {len(sr_df)} rows for {dataset_name} [{level_name}]") 
    return sr_df 

try: 
    # ========================================== 
    # --- STM_s_HTR --- 
    # ========================================== 
    df_stms_htr = pd.read_csv(stms_htr_input, keep_default_na=False) 
    df_stms_htr.columns = df_stms_htr.columns.str.strip() 
    
    stms_species_col = 'Species' if 'Species' in df_stms_htr.columns else 'Sp'
    id_col = next((c for c in ['recording', 'Filename', 'Survey_ID'] if c in df_stms_htr.columns), None) 
    
    if id_col: 
        df_stms_htr = df_stms_htr.dropna(subset=[id_col, stms_species_col])
        df_stms_htr = df_stms_htr[df_stms_htr[id_col].astype(str).str.strip() != '']

        parts_stms = df_stms_htr[id_col].astype(str).str.split('_') 
        df_stms_htr['Site'] = parts_stms.str[0] 
        
        # Drop rows where Site is blank or NaN before proceeding
        df_stms_htr = df_stms_htr[df_stms_htr['Site'].astype(str).str.strip() != '']
        df_stms_htr = df_stms_htr[df_stms_htr['Site'].astype(str).str.upper() != 'NAN']

        df_stms_htr['Date'] = parts_stms.str[-2].fillna('UnknownDate') 
        df_stms_htr['Time'] = parts_stms.str[-1].fillna('UnknownTime') 

        mask_stms = ~df_stms_htr[stms_species_col].astype(str).str.upper().str.strip().isin(['NONE NONE', 'NONE_NONE'])
        valid_stms = df_stms_htr[mask_stms].copy() 
        
        valid_stms['Site_Date'] = valid_stms['Site'].astype(str) + "_" + valid_stms['Date'].astype(str) 
        valid_stms['Site_Date_15sec'] = valid_stms['Site_Date'] + "_" + valid_stms['Time'].astype(str) 

        df_stms_htr_clean = valid_stms[['Site', 'Site_Date', 'Site_Date_15sec', stms_species_col]].copy() 
        df_stms_htr_clean.columns = ['Site', 'Site_Date', 'Site_Date_15sec', 'Species'] 
        
        # Added BEXTRA2 and BEXTRA3 to the replacement dictionary
        bextra_reps = {'BEXTRAT2': 'BExtraT2', 'BEXTRAT3': 'BExtraT3', 'BEXTRA2': 'BExtraT2', 'BEXTRA3': 'BExtraT3'}
        df_stms_htr_clean['Site'] = df_stms_htr_clean['Site'].astype(str).str.upper().replace(bextra_reps) 
        df_stms_htr_clean['entire_dataset'] = 'All_STM_s_HTR' 
        
        stms_htr_site = calculate_stm_sr(df_stms_htr_clean, 'Site', 'Site', 'STM_s_HTR')
        stms_htr_entire = calculate_stm_sr(df_stms_htr_clean, 'entire_dataset', 'entire_dataset', 'STM_s_HTR')
        stms_htr_sd = calculate_stm_sr(df_stms_htr_clean, 'Site_Date', 'Site_Date', 'STM_s_HTR')
        stms_htr_15s = calculate_stm_sr(df_stms_htr_clean, 'Site_Date_15sec', '15_Second_Interval', 'STM_s_HTR')
        
        print("  -> Reading STM_s_HTR master file list for zero-padding...")
        try:
            df_master_files = pd.read_csv(stms_master_list, header=None, names=['filename'], keep_default_na=False)
            master_keys = []
            
            for f in df_master_files['filename'].astype(str):
                f_name = f.split('.')[0].strip().upper() 
                f_parts = f_name.split('_')
                if len(f_parts) >= 3:
                    f_site_raw = f_parts[0]
                    # Apply the exact same BEXTRA map to the master list
                    for old_val, new_val in bextra_reps.items():
                        f_site_raw = f_site_raw.replace(old_val, new_val)
                    master_keys.append(f"{f_site_raw}_{f_parts[-2]}_{f_parts[-1]}")

            master_df = pd.DataFrame({'Grouping': list(set(master_keys))})
            
            stms_htr_15s['Grouping'] = stms_htr_15s['Grouping'].astype(str).str.strip().str.upper()
            stms_htr_15s = stms_htr_15s.drop_duplicates(subset=['Grouping'])
            stms_htr_15s = pd.merge(master_df, stms_htr_15s, on='Grouping', how='outer')
            
            for col in ['SR', 'SR_with_background', 'SR_with_unknowns', 'SR_with_background_and_unknowns']:
                stms_htr_15s[col] = stms_htr_15s[col].fillna(0).astype(int)
            
            stms_htr_15s['Dataset'] = stms_htr_15s['Dataset'].fillna('STM_s_HTR')
            stms_htr_15s['Summary_Level'] = stms_htr_15s['Summary_Level'].fillna('15_Second_Interval')
            print(f"  -> Successfully zero-padded STM_s_HTR. Final rows: {len(stms_htr_15s)}")
                
        except Exception as e:
            print(f"  -> WARNING: Could not read STM_s master CSV file. Error: {e}")

        all_summaries.extend([stms_htr_site, stms_htr_entire, stms_htr_sd, stms_htr_15s])

    # ========================================== 
    # --- STM_s_LTR --- 
    # ========================================== 
    if os.path.exists(stms_ltr_input):
        df_stms_ltr = pd.read_csv(stms_ltr_input, keep_default_na=False)
        
        mask_ltr = ~df_stms_ltr['Sp'].astype(str).str.upper().str.strip().isin(['NONE NONE', 'NONE_NONE'])
        valid_ltr = df_stms_ltr[mask_ltr].dropna(subset=['Sp', 'Site', 'Date', 'Time']).copy()
        
        # Drop blank sites
        valid_ltr = valid_ltr[valid_ltr['Site'].astype(str).str.strip() != '']
        
        # Apply comprehensive BEXTRA replacements
        valid_ltr['Site'] = valid_ltr['Site'].astype(str).str.upper().replace(bextra_reps)
        
        # Retain Date/Time fixes
        valid_ltr['Date'] = pd.to_datetime(valid_ltr['Date'].astype(str).str.zfill(6), format='%m%d%y').dt.strftime('%Y%m%d')
        raw_time = pd.to_datetime(valid_ltr['Time'].astype(str).str.zfill(6), format='%H%M%S', errors='coerce')
        valid_ltr['Time'] = raw_time.dt.floor('15min').dt.strftime('%H%M%S')
        
        valid_ltr['Site_Date'] = valid_ltr['Site'] + "_" + valid_ltr['Date']
        valid_ltr['Site_Date_15min'] = valid_ltr['Site_Date'] + "_" + valid_ltr['Time']
        valid_ltr['entire_dataset'] = 'All_STM_s_LTR'
        
        df_stms_ltr_clean = valid_ltr.rename(columns={'Sp': 'Species'})
        
        stms_ltr_site = calculate_stm_sr(df_stms_ltr_clean, 'Site', 'Site', 'STM_s_LTR')
        stms_ltr_entire = calculate_stm_sr(df_stms_ltr_clean, 'entire_dataset', 'entire_dataset', 'STM_s_LTR')
        stms_ltr_sd = calculate_stm_sr(df_stms_ltr_clean, 'Site_Date', 'Site_Date', 'STM_s_LTR')
        stms_ltr_15m = calculate_stm_sr(df_stms_ltr_clean, 'Site_Date_15min', '15_Minute_Interval', 'STM_s_LTR')
        
        all_summaries.extend([stms_ltr_site, stms_ltr_entire, stms_ltr_sd, stms_ltr_15m])
    else:
        print(f"  -> WARNING: Could not find STM_s_LTR input at {stms_ltr_input}")
        df_stms_ltr_clean = pd.DataFrame(columns=['Site', 'Species']) 

        
    # ========================================== 
    # --- STM_h --- 
    # ========================================== 
    df_stmh = pd.read_csv(stmh_input, keep_default_na=False) 
    stmh_species_col = 'Species' 
    
    stmh_1m_rows = []
    stmh_15s_rows = []
    
    for survey_id, group in df_stmh.groupby('Survey_ID'):
        survey_id_str = str(survey_id)
        parts = survey_id_str.split('_')
        
        raw_site = parts[0]
        # Skip processing entirely if the site is missing
        if str(raw_site).strip() == '' or str(raw_site).upper() == 'NAN': 
            continue
            
        site = raw_site.replace('.150m', '').replace('.0m', '')
        date_str = parts[-2]
        base_time_str = parts[-1]
        
        site_date = f"{site}_{date_str}"
        site_date_1min = f"{site_date}_{base_time_str}"
        
        try:
            base_time_obj = datetime.strptime(base_time_str, "%H%M%S")
        except ValueError:
            continue
            
        valid_group = group[~group[stmh_species_col].astype(str).str.upper().str.strip().isin(['NO LABELS', 'NO_LABELS', 'NOTHING'])]
        
        sr_1m = len(get_filtered_stm_species_set(valid_group[stmh_species_col], include_unknowns=False))
        sr_1m_u = len(get_filtered_stm_species_set(valid_group[stmh_species_col], include_unknowns=True))
        
        stmh_1m_rows.append({
            'Dataset': 'STM_h', 'Summary_Level': '1_Minute_Interval', 'Grouping': site_date_1min,
            'SR': sr_1m, 'SR_with_background': sr_1m, 'SR_with_unknowns': sr_1m_u, 'SR_with_background_and_unknowns': sr_1m_u
        })
        
        for i in range(4):
            rel_start = i * 15
            rel_end = (i + 1) * 15
            segment_time_obj = base_time_obj + timedelta(seconds=rel_start)
            start_time_hhmmss = segment_time_obj.strftime("%H%M%S")
            grouping_15s = f"{site_date}_{start_time_hhmmss}"
            
            overlap = valid_group[
                (valid_group['BEGIN.TIME..S.'] < rel_end) & 
                (valid_group['END.TIME..S.'] > rel_start)
            ]
            
            sr_15s = len(get_filtered_stm_species_set(overlap[stmh_species_col], include_unknowns=False))
            sr_15s_u = len(get_filtered_stm_species_set(overlap[stmh_species_col], include_unknowns=True))
            
            stmh_15s_rows.append({
                'Dataset': 'STM_h', 'Summary_Level': '15_Second_Interval', 'Grouping': grouping_15s,
                'SR': sr_15s, 'SR_with_background': sr_15s, 'SR_with_unknowns': sr_15s_u, 'SR_with_background_and_unknowns': sr_15s_u
            })
            
    df_stmh_1m = pd.DataFrame(stmh_1m_rows)
    df_stmh_15s = pd.DataFrame(stmh_15s_rows)
    
    all_summaries.append(df_stmh_1m)
    all_summaries.append(df_stmh_15s)
    print(f"  -> Generated {len(df_stmh_1m)} rows for STM_h [1_Minute_Interval]")
    print(f"  -> Generated {len(df_stmh_15s)} rows for STM_h [15_Second_Interval]")

    df_stmh['Site'] = df_stmh['Survey_ID'].astype(str).str.split('_').str[0] 
    
    # Ensure blank sites are purged from the high-level rollups
    df_stmh = df_stmh[df_stmh['Site'].astype(str).str.strip() != '']
    df_stmh = df_stmh[df_stmh['Site'].astype(str).str.upper() != 'NAN']
    
    df_stmh['Site'] = df_stmh['Site'].str.replace('.150m', '', regex=False).str.replace('.0m', '', regex=False) 

    invalid_labels = ['NO LABELS', 'NOTHING'] 
    mask_stmh = ~df_stmh[stmh_species_col].astype(str).str.upper().str.strip().isin(invalid_labels) 
    valid_stmh = df_stmh[mask_stmh].dropna(subset=[stmh_species_col]).copy() 
    
    df_stmh_clean = valid_stmh.rename(columns={stmh_species_col: 'Species'})
    parts_valid = df_stmh_clean['Survey_ID'].astype(str).str.split('_') 
    df_stmh_clean['Site_Date'] = df_stmh_clean['Site'].astype(str) + "_" + parts_valid.str[-2].fillna('UnknownDate')
    df_stmh_clean['entire_dataset'] = 'All_STM_h' 
    
    all_summaries.append(calculate_stm_sr(df_stmh_clean, 'Site', 'Site', 'STM_h')) 
    all_summaries.append(calculate_stm_sr(df_stmh_clean, 'entire_dataset', 'entire_dataset', 'STM_h')) 
    all_summaries.append(calculate_stm_sr(df_stmh_clean, 'Site_Date', 'Site_Date', 'STM_h')) 


    # ========================================== 
    # --- COMBINED DATASET ROLLUPS --- 
    # ========================================== 
    if 'df_stms_htr_clean' in locals(): 
        # For combined datasets, we use regex to blanket-replace all BEXTRA variations
        regex_rep_1 = r'(?i)BEXTRA[T]?2'
        regex_rep_2 = r'(?i)BEXTRA[T]?3'
        
        combined_stm_all_15s = pd.concat([df_stms_htr_clean[['Site', 'Species']], df_stmh_clean[['Site', 'Species']]], ignore_index=True) 
        combined_stm_all_15s['Site'] = combined_stm_all_15s['Site'].str.replace(regex_rep_1, 'BExtraT2', regex=True).str.replace(regex_rep_2, 'BExtraT3', regex=True)
        combined_stm_all_15s['entire_dataset'] = 'All_STM_15s_Combined' 
        
        all_summaries.append(calculate_stm_sr(combined_stm_all_15s, 'Site', 'Site', 'STM_All_15s')) 
        all_summaries.append(calculate_stm_sr(combined_stm_all_15s, 'entire_dataset', 'entire_dataset', 'STM_All_15s')) 

        if not df_stms_ltr_clean.empty:
            combined_stm_s_all = pd.concat([df_stms_htr_clean[['Site', 'Species']], df_stms_ltr_clean[['Site', 'Species']]], ignore_index=True)
            combined_stm_s_all['Site'] = combined_stm_s_all['Site'].str.replace(regex_rep_1, 'BExtraT2', regex=True).str.replace(regex_rep_2, 'BExtraT3', regex=True)
            combined_stm_s_all['entire_dataset'] = 'All_STM_s_Combined' 
            
            all_summaries.append(calculate_stm_sr(combined_stm_s_all, 'Site', 'Site', 'STM_s_All')) 
            all_summaries.append(calculate_stm_sr(combined_stm_s_all, 'entire_dataset', 'entire_dataset', 'STM_s_All')) 

        combined_stm_all = pd.concat([df_stms_htr_clean[['Site', 'Species']], df_stmh_clean[['Site', 'Species']], df_stms_ltr_clean[['Site', 'Species']]], ignore_index=True)
        combined_stm_all['Site'] = combined_stm_all['Site'].str.replace(regex_rep_1, 'BExtraT2', regex=True).str.replace(regex_rep_2, 'BExtraT3', regex=True)
        combined_stm_all['entire_dataset'] = 'All_STM_Combined' 
        
        all_summaries.append(calculate_stm_sr(combined_stm_all, 'Site', 'Site', 'STM_All')) 
        all_summaries.append(calculate_stm_sr(combined_stm_all, 'entire_dataset', 'entire_dataset', 'STM_All')) 

except Exception as e: 
    print(f"  -> ERROR processing STM datasets: {e}") 

# ========================================== 
# 4. COMPILE AND EXPORT 
# ========================================== 
print("\n=== COMPILING FINAL SPREADSHEET ===") 
valid_summaries = [s for s in all_summaries if s is not None] 

if len(valid_summaries) > 0: 
    df_final = pd.concat(valid_summaries, ignore_index=True) 
    final_columns = [ 
        'Dataset', 'Summary_Level', 'Grouping',  
        'SR', 'SR_with_background', 'SR_with_unknowns', 'SR_with_background_and_unknowns' 
    ] 
    df_final = df_final[final_columns] 

    os.makedirs(output_dir, exist_ok=True) 
    df_final.to_csv(output_csv, index=False) 
    print(f"Success! Final Master Summary saved to: {output_csv}") 
    print(f"Total Rows Saved: {len(df_final)}") 
else: 
    print("ERROR: No summaries were successfully generated. Nothing saved.")
