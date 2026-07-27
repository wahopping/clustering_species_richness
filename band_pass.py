import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

# --- Configuration ---
INPUT_DIR = '[path for full_range files]'
OUTPUT_DIR = '[path for bird_range outputs]'

# Standard Bioacoustics Parameters
LOW_CUT = 500.0   
HIGH_CUT = 4000.0 
FILTER_ORDER = 9 

def process_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.wav')]
    print(f"Found {len(files)} files. Processing...")

    for filename in files:
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # Read file
            fs, data = wavfile.read(in_path)

            # Design filter specific to this file's sample rate (fs)
            nyq = 0.5 * fs
            low = LOW_CUT / nyq
            high = HIGH_CUT / nyq
            sos = butter(FILTER_ORDER, [low, high], btype='band', output='sos')

            # Apply filter (handle stereo vs mono)
            # Apply filter (Forward-Backward for sharpest cut)
            # sosfiltfilt handles the dimensions automatically better than sosfilt
            if len(data.shape) > 1:
                filtered_data = sosfiltfilt(sos, data, axis=0)
            else:
                filtered_data = sosfiltfilt(sos, data, axis=-1)

            # Save file - Cast back to original data type (e.g., int16), this ensures the volume/format matches the original file exactly.
            wavfile.write(out_path, fs, filtered_data.astype(data.dtype))
            
            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_files()
