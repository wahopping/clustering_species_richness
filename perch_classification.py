#Running Perch v2 (and v1, commented out) for classification
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
MODEL_URL = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'
# MODEL_URL = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/1' #for perch 1.0

# --- 1. SET THIS TO TRUE FOR YOUR FIRST RUN ---
CALIBRATION_MODE = False

# When calibrating, we drop raw logits below this number to keep file sizes small 
# and prevent OOM crashes on the cluster.
RAW_LOGIT_CUTOFF = 1.0

# --- 2. SET THESE AFTER CALIBRATION, THEN SET CALIBRATION_MODE = FALSE ---
CALIBRATED_OFFSET = 10.0482     
CALIBRATED_TEMPERATURE = 0.7346 
PROBABILITY_CUTOFF = 0.05

TEST_RUN = False
CALIBRATION_POWER = 8 
WRITE_BATCH_SIZE = 250 # Reduced slightly to ensure we stay well under OOM limits

def setup_gpu_memory():
    """Prevents TF from pre-allocating 100% of GPU memory."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth error: {e}")

# Original Perch 1.0 Sigmoid Function (use when running Perch 1.0)
# def safe_sigmoid(x):
#     x = np.clip(x, -100, 100)
#     return 1 / (1 + np.exp(-x))

def calibrated_sigmoid(x, offset=CALIBRATED_OFFSET, temperature=CALIBRATED_TEMPERATURE):
    """Applies your customized temperature and offset scaling to Perch v2.0 logits."""
    # Explicitly cast to float64 to prevent exp overflow warnings
    scaled_x = (x.astype(np.float64) - offset) / temperature
    scaled_x = np.clip(scaled_x, -700, 700)
    return 1 / (1 + np.exp(-scaled_x))

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
    """Processes audio and extracts logits or calibrated probabilities."""
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
        
        # --- Dynamically identify the true logits tensor ---
        # Perch v2 logits have ~14,800 classes, while embeddings only have 1536
        logits_tensor = None
        for key, tensor in outputs.items():
            if tensor.shape[-1] > 2000:
                logits_tensor = tensor
                break
                
        if logits_tensor is None:
            raise ValueError(f"Could not find logits in model outputs! Keys found: {list(outputs.keys())}")
            
        all_window_logits.append(logits_tensor.numpy()[0])
            
    max_logits = np.max(all_window_logits, axis=0)
    file_results = []
    
    # --- PROCESSING MODES ---
    if CALIBRATION_MODE:
        # Save only strong raw logits to keep the calibration files tiny
        top_indices = np.where(max_logits > RAW_LOGIT_CUTOFF)[0]
        
        for idx in top_indices:
            raw_logit = max_logits[idx]
            species_name = sci_labels[idx] if idx < len(sci_labels) else "unknown"
            
            file_results.append({
                'recording': recording,
                'species': species_name,
                'raw_logit': raw_logit,
                'calibrated_probability': np.nan, 
                'power_calibrated': np.nan
            })
            
    else:
        # Standard Inference Mode
        calibrated_probs = calibrated_sigmoid(max_logits)
        top_indices = np.where(calibrated_probs > PROBABILITY_CUTOFF)[0]
        
        for idx in top_indices:
            raw_logit = max_logits[idx]
            probs = calibrated_probs[idx]
            species_name = sci_labels[idx] if idx < len(sci_labels) else "unknown"
            
            file_results.append({
                'recording': recording,
                'species': species_name,
                'raw_logit': raw_logit,
                'calibrated_probability': probs,
                'power_calibrated': probs ** CALIBRATION_POWER
            })

    # Clear massive arrays from memory immediately
    del audio, all_window_logits, max_logits, outputs
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
    if CALIBRATION_MODE:
        print(f"CALIBRATION MODE ON: Only saving raw logits > {RAW_LOGIT_CUTOFF}")
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    empty_df = pd.DataFrame(columns=['recording', 'species', 'raw_logit', 'calibrated_probability', 'power_calibrated'])
    empty_df.to_csv(args.output_csv, index=False)
    
    current_batch = []
    
    for count, filepath in enumerate(audio_files, 1):
        res = process_audio_file(filepath, model, sci_labels)
        if res: 
            current_batch.extend(res)
            
        if count % WRITE_BATCH_SIZE == 0:
            print(f"Processed {count}/{len(audio_files)} files. Writing batch to disk...")
            if current_batch:
                df = pd.DataFrame(current_batch)
                df.to_csv(args.output_csv, mode='a', header=False, index=False)
            
            current_batch = []
            gc.collect()
            
    if current_batch:
        df = pd.DataFrame(current_batch)
        df.to_csv(args.output_csv, mode='a', header=False, index=False)
        
    print(f"Complete. Results successfully written to {args.output_csv}")

if __name__ == "__main__":
    main()
