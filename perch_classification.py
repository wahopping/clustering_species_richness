#Running Perch v2 for classification

import os
import glob
import argparse
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# --- Configuration ---
SAMPLE_RATE = 32000
WINDOW_SIZE = 5 * SAMPLE_RATE

# Perch v2 Model URL
MODEL_URL = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'

# Perch v1: https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8

# Set to True to limit processing to 50 files for debugging/testing
TEST_RUN = False

# We keep this for reference, but you can rely on the raw logits moving forward
CALIBRATION_POWER = 8 

# Drop the "long tail" of ~9,900 species that score near zero to save CSV space.
LOGIT_CUTOFF = -5.0 

# How many files to process before writing to disk and clearing RAM
WRITE_BATCH_SIZE = 500 

def setup_gpu_memory():
    """Prevents TF from pre-allocating 100% of GPU memory, avoiding fragmentation OOMs."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth error: {e}")

def safe_sigmoid(x):
    """Safely applies sigmoid to avoid math overflow warnings."""
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def load_perch_labels():
    """Extracts the official species taxonomy directly from the Kaggle model assets."""
    model_path = hub.resolve(MODEL_URL)
    
    csv_path = os.path.join(model_path, "assets", "label.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(model_path, "assets", "labels.csv")
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find taxonomy CSV in {model_path}/assets/")
    
    df = pd.read_csv(csv_path)
    
    sci_col = [c for c in df.columns if 'sci' in c.lower() or 'ebird2021' in c.lower()]
    if sci_col:
        return df[sci_col[0]].astype(str).str.lower().str.strip().tolist()
    else:
        return df.iloc[:, 0].astype(str).str.lower().str.strip().tolist()

def process_audio_file(filepath, model, sci_labels):
    """Processes audio and extracts all raw logits above the cutoff."""
    filename = os.path.basename(filepath)
    recording = filename.replace('.wav', '')
    
    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

    if len(audio) % WINDOW_SIZE != 0:
        pad_length = WINDOW_SIZE - (len(audio) % WINDOW_SIZE)
        audio = np.pad(audio, (0, pad_length))

    num_windows = len(audio) // WINDOW_SIZE
    all_window_logits = []
    
    infer = model.signatures['serving_default']
    
    # Run Native Perch Inference on 5s chunks
    for i in range(num_windows):
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE
        
        window = audio[start_idx:end_idx].astype(np.float32)
        window_batched = window[np.newaxis, :]
        
        outputs = infer(inputs=tf.constant(window_batched))
        logits = outputs.get('output_0', list(outputs.values())[0])
        
        all_window_logits.append(logits.numpy()[0])
            
    # Take the highest logit for each species across the file
    max_logits = np.max(all_window_logits, axis=0)
    
    file_results = []
    top_indices = np.where(max_logits > LOGIT_CUTOFF)[0]
    
    for idx in top_indices:
        raw_logit = max_logits[idx]
        species_name = sci_labels[idx] if idx < len(sci_labels) else "unknown"
        
        probs = safe_sigmoid(raw_logit)
        power_calibrated = probs ** CALIBRATION_POWER
        
        file_results.append({
            'recording': recording,
            'species': species_name,
            'raw_logit': raw_logit,
            'standard_sigmoid': probs,
            'power_calibrated': power_calibrated
        })

    return file_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    setup_gpu_memory()

    print("Loading Google Perch Model and Taxonomy...")
    model = hub.load(MODEL_URL)
    sci_labels = load_perch_labels()

    audio_files = glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
    
    if TEST_RUN:
        print("TEST RUN ENABLED: Limiting processing to 50 files.")
        audio_files = audio_files[:50]
        
    print(f"Found {len(audio_files)} audio files to process.")
    
    # Initialize the output CSV file and write the header
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    empty_df = pd.DataFrame(columns=['recording', 'species', 'raw_logit', 'standard_sigmoid', 'power_calibrated'])
    empty_df.to_csv(args.output_csv, index=False)
    
    current_batch = []
    
    for count, filepath in enumerate(audio_files, 1):
        res = process_audio_file(filepath, model, sci_labels)
        if res: 
            current_batch.extend(res)
            
        # Write to disk and clear RAM every WRITE_BATCH_SIZE files
        if count % WRITE_BATCH_SIZE == 0:
            print(f"Processed {count}/{len(audio_files)} files. Writing batch to disk...")
            if current_batch:
                df = pd.DataFrame(current_batch)
                # mode='a' appends to the CSV without overwriting
                df.to_csv(args.output_csv, mode='a', header=False, index=False)
            
            # Clear the batch from RAM and force garbage collection
            current_batch = []
            gc.collect()
            
    # Save any remaining files in the final partial batch
    if current_batch:
        df = pd.DataFrame(current_batch)
        df.to_csv(args.output_csv, mode='a', header=False, index=False)
        
    print(f"Complete. Detailed logits successfully written to {args.output_csv}")

if __name__ == "__main__":
    main()
