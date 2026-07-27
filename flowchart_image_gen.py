#script for making components of flowchart

#!/usr/bin/env python3
"""
flowchart_image_gen.py
==============================
Generates four flowchart PNGs:
1. 1_min_split.png: 1-minute file split into four 15s chunks with jagged outer edges (Tall, up to 22.5kHz).
2. band_pass_image.png: Full range with broken axis at the top.
3. separated_stack.png: Solid diagonal stack, spaced out.
4. separated_grid.png: 2x4 grid of the 8 separated channels.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
except ImportError:
    print("ERROR: librosa is required.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Directories & Parameters
# ---------------------------------------------------------------------------
TARGET_STEM = "Site02_20190131_054800"

ONE_MIN_DIR = Path("[root]/ground_truth/full_range/PER_1_min")
FULL_RANGE_DIR = Path("[root]/ground_truth/full_range/PER_15s")
SEPARATED_DIR = Path("[root]/ground_truth/bird_range/PER/separated")
ACTIVE_DIR = Path("[root]/ground_truth/bird_range/PER/active")
OUTPUT_DIR = Path("[root]/flowchart/samples")

# STFT Settings
N_FFT = 1024
OVERLAP = 0.75
HOP_LENGTH = int(N_FFT * (1.0 - OVERLAP))

# Display Settings
FMIN_BIRD = 500
FMAX_BIRD = 4000
FMAX_1MIN = 22500            # Extended top range for the 1-minute split image
FMAX_DISPLAY_BOT = 10500     # Cutoff for the bottom part of the band-pass broken axis
FMAX_DISPLAY_TOP_MIN = 14000 # Resume axis here (Band-pass)
FMAX_DISPLAY_TOP_MAX = 15000 # Top of the Y axis (Band-pass)

# Visual threshold
VMIN = -55 
VMAX = 0
OPACITY = 1.0


# ---------------------------------------------------------------------------
# Audio & STFT Helpers
# ---------------------------------------------------------------------------
def load_audio(wav_path: Path):
    if SOUNDFILE_AVAILABLE:
        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            return y, int(sr)
        except Exception:
            pass
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        return None, None


def compute_spectrogram(wav_path: Path, max_freq=4000):
    y, sr = load_audio(wav_path)
    if y is None or len(y) == 0:
        return None, None, None, None
    
    S = np.abs(librosa.stft(
        y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True
    )) ** 2
    
    S_db = librosa.power_to_db(S, ref=1.0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    duration_s = len(y) / sr
    
    bin_hi = int(np.searchsorted(freqs, max_freq, side="right"))
    bin_hi = min(bin_hi, S_db.shape[0] - 1)
    
    return S_db[:bin_hi, :], freqs[:bin_hi], duration_s, sr


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print(f"Targeting recording: {TARGET_STEM}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    one_min_wav = ONE_MIN_DIR / f"{TARGET_STEM}.wav"
    full_wav = FULL_RANGE_DIR / f"{TARGET_STEM}.wav"
    
    if not full_wav.exists():
        print(f"ERROR: Could not find {full_wav}")
        sys.exit(1)
        
    sep_files = sorted(SEPARATED_DIR.glob(f"{TARGET_STEM}*.wav"))
    if len(sep_files) != 8:
        print(f"ERROR: Found {len(sep_files)} separated channels, expected 8.")
        sys.exit(1)


    # ==========================================================
    # ASSET 1: 1-Minute File Split into 15s Segments (Taller)
    # ==========================================================
    if one_min_wav.exists():
        print("Generating 1/4: 1-minute split illustration...")
        # Increased height to 7.5 to make it notably taller
        fig_1m, axs = plt.subplots(1, 4, figsize=(16, 7.5), facecolor="white", 
                                   sharey=True, gridspec_kw={'wspace': 0.1})
        
        S_db_1m, freqs_1m, dur_1m, sr = compute_spectrogram(one_min_wav, max_freq=FMAX_1MIN)
        
        # Calculate frames for 15s chunks
        frames_per_sec = S_db_1m.shape[1] / dur_1m
        frames_15s = int(15 * frames_per_sec)
        actual_fmax = freqs_1m[-1]
        
        # Jagged edge math (scaled teeth to match the taller Y-axis)
        y_jag = np.linspace(0, actual_fmax, 200)
        jag_depth = 0.5  # Width of the zig-zag in seconds
        jag_pattern = jag_depth * np.abs((y_jag % 3000) - 1500) / 1500
        
        for i in range(4):
            ax = axs[i]
            start_f = i * frames_15s
            end_f = (i + 1) * frames_15s if i < 3 else S_db_1m.shape[1]
            S_chunk = S_db_1m[:, start_f:end_f]
            
            ax.imshow(
                S_chunk, origin="lower", aspect="auto", cmap="magma",
                vmin=VMIN, vmax=VMAX, extent=[0, 15, freqs_1m[0], actual_fmax]
            )
            
            ax.set_xticks([0, 5, 10, 15])
            ax.set_xticklabels([f"{i*15}", f"{i*15+5}", f"{i*15+10}", f"{i*15+15}"])
            ax.set_xlabel("Time (s)")
            
            # Leftmost jagged edge
            if i == 0:
                ax.fill_betweenx(y_jag, 0, jag_pattern, color='white', zorder=10)
                ax.plot(jag_pattern, y_jag, color='black', lw=1.5, zorder=11)
                ax.spines['left'].set_visible(False)
            
            # Rightmost jagged edge
            if i == 3:
                ax.fill_betweenx(y_jag, 15 - jag_pattern, 15, color='white', zorder=10)
                ax.plot(15 - jag_pattern, y_jag, color='black', lw=1.5, zorder=11)
                ax.spines['right'].set_visible(False)

        axs[0].set_ylabel("Frequency (Hz)")
        fig_1m.suptitle(f"Continuous Recording Chunking\n{TARGET_STEM} (1 min)", fontweight="bold")
        
        out_1m = OUTPUT_DIR / f"1_min_split_{TARGET_STEM}.png"
        fig_1m.savefig(out_1m, dpi=200, bbox_inches="tight")
        plt.close(fig_1m)
    else:
        print(f"Skipping 1/4: Could not find 1-min file at {one_min_wav}")


    # ==========================================================
    # ASSET 2: Band-Pass Illustration (Broken Axis)
    # ==========================================================
    print("Generating 2/4: Band-pass illustration...")
    fig_bp = plt.figure(figsize=(6, 8), facecolor="white")
    gs_bp = gridspec.GridSpec(2, 1, height_ratios=[1, 8], hspace=0.08)
    ax_top = fig_bp.add_subplot(gs_bp[0])
    ax_bot = fig_bp.add_subplot(gs_bp[1])
    
    S_db_full, freqs_full, dur_full, _ = compute_spectrogram(full_wav, max_freq=FMAX_DISPLAY_TOP_MAX)
    
    b500 = int(np.searchsorted(freqs_full, FMIN_BIRD))
    b4000 = int(np.searchsorted(freqs_full, FMAX_BIRD))
    
    # Bottom Axis Plotting (0 to 10500 Hz)
    ext_low = [0, dur_full, freqs_full[0], freqs_full[b500]]
    ext_mid = [0, dur_full, freqs_full[b500], freqs_full[b4000]]
    ext_high = [0, dur_full, freqs_full[b4000], freqs_full[-1]]
    
    ax_bot.imshow(S_db_full[:b500, :], origin="lower", aspect="auto", cmap="gray",
                 vmin=VMIN, vmax=VMAX, extent=ext_low)
    ax_bot.imshow(S_db_full[b500:b4000, :], origin="lower", aspect="auto", cmap="magma",
                 vmin=VMIN, vmax=VMAX, extent=ext_mid)
    ax_bot.imshow(S_db_full[b4000:, :], origin="lower", aspect="auto", cmap="gray",
                 vmin=VMIN, vmax=VMAX, extent=ext_high)
                 
    ax_bot.axhline(FMIN_BIRD, color='red', linestyle='--', linewidth=2)
    ax_bot.axhline(FMAX_BIRD, color='red', linestyle='--', linewidth=2)
    ax_bot.set_ylim(0, FMAX_DISPLAY_BOT)
    ax_bot.set_xlim(0, dur_full)
    
    # Top Axis Plotting (14000 to 15000 Hz)
    ax_top.imshow(S_db_full, origin="lower", aspect="auto", cmap="gray",
                 vmin=VMIN, vmax=VMAX, extent=[0, dur_full, freqs_full[0], freqs_full[-1]])
    ax_top.set_ylim(FMAX_DISPLAY_TOP_MIN, FMAX_DISPLAY_TOP_MAX)
    ax_top.set_xlim(0, dur_full)
    
    # Hide spines to create the "break"
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    
    # Draw standard diagonal break lines
    d = .015 
    kwargs = dict(transform=ax_top.transAxes, color='black', clip_on=False, lw=1.5)
    ax_top.plot((-d, +d), (-d*4, +d*4), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d*4, +d*4), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    ax_bot.set_xlabel("Time (s)", fontsize=10)
    fig_bp.text(0.02, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize=10)
    ax_top.set_title(f"Band-pass Filter\n{TARGET_STEM}", fontsize=12, pad=10)
    
    out_bp = OUTPUT_DIR / f"band_pass_image_{TARGET_STEM}.png"
    fig_bp.savefig(out_bp, dpi=200, bbox_inches="tight")
    plt.close(fig_bp)


    # ==========================================================
    # ASSET 3: Separated Stack (Solid)
    # ==========================================================
    print("Generating 3/4: Separated channel stack...")
    fig_stack = plt.figure(figsize=(10, 8), facecolor="none")
    
    rect_w, rect_h = 0.4, 0.3
    shift_x, shift_y = 0.075, 0.085 
    
    for i in range(7, -1, -1):
        sep_wav = sep_files[i]
        S_db_sep, freqs_sep, dur_sep, _ = compute_spectrogram(sep_wav, max_freq=FMAX_BIRD)
        
        b500 = int(np.searchsorted(freqs_sep, FMIN_BIRD))
        S_db_sep = S_db_sep[b500:, :]
        freqs_sep = freqs_sep[b500:]
        
        is_active = (ACTIVE_DIR / sep_wav.name).exists()
        color = "#2ca02c" if is_active else "#d62728"
        
        x_pos = 0.05 + (i * shift_x)
        y_pos = 0.05 + (i * shift_y)
        
        ax_stack = fig_stack.add_axes([x_pos, y_pos, rect_w, rect_h])
        ax_stack.imshow(
            S_db_sep, origin="lower", aspect="auto", interpolation="none",
            cmap="magma", vmin=VMIN, vmax=VMAX, 
            extent=[0, dur_sep, freqs_sep[0], freqs_sep[-1]], 
            alpha=OPACITY, zorder=3
        )
        ax_stack.set_xticks([])
        ax_stack.set_yticks([])
        
        for spine in ax_stack.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(8.0) 
            spine.set_zorder(2)

    out_stack = OUTPUT_DIR / f"separated_stack_{TARGET_STEM}.png"
    fig_stack.savefig(out_stack, dpi=200, transparent=True)
    plt.close(fig_stack)


    # ==========================================================
    # ASSET 4: Separated Grid
    # ==========================================================
    print("Generating 4/4: Separated channel grid...")
    fig_grid = plt.figure(figsize=(16, 6), facecolor="white")
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.15)
    
    for i, sep_wav in enumerate(sep_files):
        row = i // 4
        col = i % 4
        
        ax_grid = fig_grid.add_subplot(gs[row, col])
        S_db_sep, freqs_sep, dur_sep, _ = compute_spectrogram(sep_wav, max_freq=FMAX_BIRD)
        
        b500 = int(np.searchsorted(freqs_sep, FMIN_BIRD))
        S_db_sep = S_db_sep[b500:, :]
        freqs_sep = freqs_sep[b500:]
        
        is_active = (ACTIVE_DIR / sep_wav.name).exists()
        color = "#2ca02c" if is_active else "#d62728"
        label = "Active" if is_active else "Noise"
        
        ax_grid.imshow(
            S_db_sep, origin="lower", aspect="auto", interpolation="none",
            cmap="magma", vmin=VMIN, vmax=VMAX, 
            extent=[0, dur_sep, freqs_sep[0], freqs_sep[-1]], 
            zorder=3
        )
        
        ax_grid.set_title(f"Channel {i+1} [{label}]", color=color, fontweight="bold", fontsize=11)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        for spine in ax_grid.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(7.0) 
            spine.set_zorder(2)
            
    out_grid = OUTPUT_DIR / f"separated_grid_{TARGET_STEM}.png"
    fig_grid.savefig(out_grid, dpi=300, bbox_inches="tight")
    plt.close(fig_grid)

    print("\nSuccess! All four PNGs generated.")

if __name__ == "__main__":
    main()
