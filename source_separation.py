#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import librosa
from scipy.io import wavfile
import tensorflow as tf
from pathlib import Path
import warnings

# --- CONFIGURATION ---
BASE_INPUT_DIR = "[path to where 15 second soundfiles are stored]" #using base in storage, no reason to copy
BASE_OUTPUT_DIR = "[path to output for separated files]" #fr_stm_s only
CHECKPOINT_PATH = "[path to model checkpoint folder in HPC]/mixit/bird_mixit_model/bird_mixit_models/bird_mixit_model_checkpoints/output_sources8"
SUB_DIRS = ["full_range", "bird_range"]
DATASETS = ["STM_soft", "STM_hard", "PER", "STM_s_LTR"]
TARGET_SR = 22050
NUM_SOURCES = 8
CHUNK_DURATION = 5.0 


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_mixit_from_checkpoint(checkpoint_path):
    meta_file = Path(checkpoint_path) / "inference.meta"
    if not meta_file.exists(): raise FileNotFoundError(f"inference.meta not found")
    
    ckpt_files = list(Path(checkpoint_path).glob("*.ckpt-*.index"))
    if not ckpt_files: raise FileNotFoundError(f"No checkpoint index found")
    ckpt_prefix = str(ckpt_files[0]).replace('.index', '')
    
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(str(meta_file))
        input_tensor = graph.get_tensor_by_name('input_audio/receiver_audio:0')
        output_tensor = graph.get_tensor_by_name('denoised_waveforms:0')

    sess = tf.compat.v1.Session(graph=graph)
    with graph.as_default():
        saver.restore(sess, ckpt_prefix)
    return sess, input_tensor, output_tensor

def process_file(file_path, output_dir, sess, input_tensor, output_tensor):
    filename = Path(file_path).stem

    #1: Load Audio
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception: return

    if audio.dtype != np.float32: audio = audio.astype(np.float32)

    #2: Process in chunks

    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    total_samples = len(audio)
    separated_streams = [[] for _ in range(NUM_SOURCES)]
    
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
        
        try:
            result = sess.run(output_tensor, feed_dict={input_tensor: chunk[np.newaxis, np.newaxis, :]})
            result = np.squeeze(result)
            if result.shape[0] > result.shape[1]: result = result.T
            for i in range(NUM_SOURCES): separated_streams[i].append(result[i])
        except Exception: continue

    #3: Save files

    for i in range(NUM_SOURCES):
        if separated_streams[i]:
            full = np.concatenate(separated_streams[i])[:total_samples] # Trim padding
            wavfile.write(os.path.join(output_dir, f"{filename}_source{i}.wav"), TARGET_SR, full)

def main():
    # Parse arguments for Array processing
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="Current SLURM task ID")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of tasks")
    args = parser.parse_args()

    # 1. Collect ALL files first
    all_files = []
    
    # Nested loop to handle the new directory layers
    for sub_dir in SUB_DIRS:
        for dataset in DATASETS:
            input_dir = os.path.join(BASE_INPUT_DIR, sub_dir, dataset)
            output_dir = os.path.join(BASE_OUTPUT_DIR, sub_dir, dataset, "separated")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get files and store tuple (input_path, output_dir)
            files = glob.glob(os.path.join(input_dir, "*.wav"))
            for f in files:
                all_files.append((f, output_dir))

    # Sort to ensure every job agrees on the order
    all_files.sort()
    
    # 2. Slice the list for THIS specific job
    my_files = all_files[args.task_id::args.num_tasks]
    
    print(f"Task {args.task_id}/{args.num_tasks} starting.")
    print(f"Processing {len(my_files)} files out of {len(all_files)} total.")

    # 3. Load Model
    try:
        sess, input_tensor, output_tensor = load_mixit_from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # 4. Process Loop
    for i, (f_path, out_dir) in enumerate(my_files):
        try:
            filename = Path(f_path).stem
            # Check existing output (using source0 as proxy)
            if os.path.exists(os.path.join(out_dir, f"{filename}_source0.wav")): #this prevents overwriting, can comment out if want to overwite
                continue

            # Print progress periodically
            if i % 5 == 0:
                print(f"Task {args.task_id}: Processing {filename}...")
            
            process_file(f_path, out_dir, sess, input_tensor, output_tensor)
        except Exception as e:
            print(f"Task {args.task_id} failed on {f_path}: {e}")

if __name__ == "__main__":
    main()
