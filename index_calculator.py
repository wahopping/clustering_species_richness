#Code for generating acoustic index values

import pandas as pd
from maad import sound, features, util
import numpy as np
import os
import fnmatch
from multiprocessing import Pool
import sys
import argparse
import warnings
import gc
warnings.filterwarnings('ignore')

# --- CONFIGURATION DICTIONARY ---
RUN_CONFIGS = {
    '15sec': {
        'INPUT_DIRS': ['PATHS TO 15 sec SOUND FILES; PER, STM_s_HTR, AND STM_h'],
        'OUTPUT_CSV': ['OUTPUT PATH FOR 15 s INDEX VALUES'],
        'SUMMARY_LEVEL': '15_Second_Interval',
        'CORES': 64
    },
    '1min': {
        'INPUT_DIRS': ['PATHS TO 1 MIN SOUND FILES; PER AND STM_h'],
        'OUTPUT_CSV': ['OUTPUT PATH FOR 1 MIN INDEX VALUES'],
        'SUMMARY_LEVEL': '1_Minute_Interval',
        'CORES': 64
    },
    '15min': {
        'INPUT_DIRS': ['PATHS TO 15 MIN SOUND FILES; PER and STM_s_LTR'],
        'OUTPUT_CSV': ['OUTPUT PATH FOR 15 MIN INDEX VALUES'],
        'SUMMARY_LEVEL': '15_Minute_Interval',
        'CORES': 20  # Safe number for 240GB of RAM (Allocates ~12GB max per process)
    }
}

# Frequency limits (Constant across all runs)
BR_LIMITS = (500, 4000)
FR_LIMITS = (0, 22050)

# ---------------------

def list_wav_files(directories):
    """Walks through directories and creates a list of full file paths."""
    wav_files = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist: {directory}", file=sys.stderr)
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, '*.wav') or fnmatch.fnmatch(file, '*.WAV'):
                    wav_files.append(os.path.join(root, file))
    return wav_files

def calculate_indices_from_spectrogram(Sxx, tn, fn, ext, freq_limits, prefix):
    """Calculates maad indices from a pre-computed spectrogram."""
    
    metrics = ['ACI', 'BI', 'surface_roughness', 'spectral_snr', 
               'ACTspfract_avg', 'EVNspCount_sum', 'EVNspFract_avg', 'ROIcover']
    result = {f"{prefix}{m}": 'NA' for m in metrics}
    
    try:
        # Filter spectrogram to specific frequency range
        freq_mask = (fn >= freq_limits[0]) & (fn <= freq_limits[1])
        Sxx_filtered = Sxx[freq_mask, :]
        fn_filtered = fn[freq_mask]
        ext_filtered = [ext[0], ext[1], freq_limits[0], freq_limits[1]]
        
        # ACI
        _, _, ACI = features.acoustic_complexity_index(Sxx_filtered)
        result[f'{prefix}ACI'] = ACI
        
        # BI
        BI = features.bioacoustics_index(Sxx, fn, flim=freq_limits, R_compatible='soundecology')
        result[f'{prefix}BI'] = BI
        
        # Surface Roughness
        surface_roughness = features.surface_roughness(Sxx_filtered, norm='global')
        result[f'{prefix}surface_roughness'] = sum(surface_roughness[1])
        
        # Spectral SNR
        snr = sound.spectral_snr(Sxx_filtered)
        result[f'{prefix}spectral_snr'] = snr[2]
        
        # Noise reduction for spectral activity/events/ROI
        Sxx_noNoise_filtered = sound.median_equalizer(Sxx_filtered, display=False, extent=ext_filtered)
        Sxx_dB_noNoise_filtered = util.power2dB(Sxx_noNoise_filtered)
        
        # Spectral Activity
        ACTspfract_per_bin, _, _ = features.spectral_activity(Sxx_dB_noNoise_filtered, dB_threshold=6)  
        result[f'{prefix}ACTspfract_avg'] = np.mean(ACTspfract_per_bin)
        
        # Spectral Events
        EVNspFract_per_bin, _, EVNspCount_per_bin, _ = features.spectral_events(
            Sxx_dB_noNoise_filtered, dt=tn[1] - tn[0], dB_threshold=6, 
            rejectDuration=0.05, display=False, extent=ext_filtered
        )
        result[f'{prefix}EVNspCount_sum'] = sum(EVNspCount_per_bin)
        result[f'{prefix}EVNspFract_avg'] = np.mean(EVNspFract_per_bin)
        
        # ROI Cover
        try:
            _, ROIcover = features.region_of_interest_index(
                Sxx_dB_noNoise_filtered, tn, fn_filtered, display=False, 
                smooth_param1=1, mask_mode='relative', mask_param1=6, mask_param2=0.5
            )
            result[f'{prefix}ROIcover'] = ROIcover
        except Exception:
            result[f'{prefix}ROIcover'] = 'NA'
            
    except Exception as e:
        print(f"Error processing {prefix} metrics: {e}", file=sys.stderr, flush=True)
        
    return result

def process_file(wav_file_path):
    """Worker function to process a single file for both ranges."""
    filename = os.path.basename(wav_file_path)
    row_results = {'File': filename, 'Filepath': wav_file_path}
    
    try:
        # Load audio ONCE
        s, fs = sound.load(wav_file_path)
        fs = 44100 
        
        # Compute spectrogram ONCE
        Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=512, noverlap=512/2)
        
        # Compute Bird Range 
        br_results = calculate_indices_from_spectrogram(Sxx, tn, fn, ext, BR_LIMITS, 'br_')
        row_results.update(br_results)
        
        # Compute Full Range 
        fr_results = calculate_indices_from_spectrogram(Sxx, tn, fn, ext, FR_LIMITS, 'fr_')
        row_results.update(fr_results)
        
    except Exception as e:
        # DO NOT fail silently anymore. Print directly to standard error.
        print(f"CRITICAL ERROR on file {filename}: {e}", file=sys.stderr, flush=True)
    
    finally:
        # Force garbage collection to prevent RAM bloat
        del s, Sxx
        gc.collect()
        
    return row_results

def format_for_master(df, summary_level):
    """Formats the raw results to easily merge into the Master Spreadsheet."""
    print(f"Formatting output columns for {summary_level}...")
    
    file_str = df['File'].astype(str).str.upper().str.replace('.WAV', '')
    parts = file_str.str.split('_')
    
    df['Site_Raw'] = parts.str[0]
    df['Site_Raw'] = df['Site_Raw'].str.replace('.150M', '', regex=False).str.replace('.0M', '', regex=False)
    
    bextra_reps = {'BEXTRAT2': 'BExtraT2', 'BEXTRAT3': 'BExtraT3', 'BEXTRA2': 'BExtraT2', 'BEXTRA3': 'BExtraT3'}
    df['Site'] = df['Site_Raw'].replace(bextra_reps)
    
    df['Date'] = parts.str[-2]
    df['Time'] = parts.str[-1]
    
    df = df[df['Time'].astype(str).str.len() == 6].copy()
    
    df['Grouping'] = df['Site'] + "_" + df['Date'] + "_" + df['Time']
    df['Summary_Level'] = summary_level
    
    cols_to_move = ['Summary_Level', 'Grouping', 'File']
    metric_cols = [c for c in df.columns if c.startswith('br_') or c.startswith('fr_')]
    final_cols = cols_to_move + metric_cols
    
    return df[final_cols]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate acoustic indices for different temporal resolutions.")
    parser.add_argument('resolution', choices=['15sec', '1min', '15min'], 
                        help="Temporal resolution to process (e.g., '15sec', '1min', or '15min')")
    args = parser.parse_args()
    
    config = RUN_CONFIGS[args.resolution]
    
    print(f"--- Starting {args.resolution} run ---", flush=True)
    print("Mapping audio files from directories...", flush=True)
    wav_files = list_wav_files(config['INPUT_DIRS'])
    
    if len(wav_files) == 0:
        print("No .wav files found. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(wav_files)} total files.", flush=True)
    print(f"Starting parallel processing with {config['CORES']} CPU cores...", flush=True)
    
    # maxtasksperchild=1 forces the worker process to completely restart after EVERY file, 
    # completely eliminating any possibility of a memory leak across files.
    with Pool(processes=config['CORES'], maxtasksperchild=1) as pool:
        results_list = pool.map(process_file, wav_files)
        
    print("Formatting results...", flush=True)
    results_df = pd.DataFrame(results_list)
    final_df = format_for_master(results_df, config['SUMMARY_LEVEL'])
    
    os.makedirs(os.path.dirname(config['OUTPUT_CSV']), exist_ok=True)
    final_df.to_csv(config['OUTPUT_CSV'], index=False)
    
    print(f"Done! Indices saved to: {config['OUTPUT_CSV']}", flush=True)
