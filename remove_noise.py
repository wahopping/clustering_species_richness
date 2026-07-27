#!/usr/bin/env python3
"""
remove_noise.py
====================
Classifies bird_mixit source-separated .wav files as noise or not-noise,
copying them into "noise/" and "active/" sibling folders alongside
each "separated/" directory.

THRESHOLD PHILOSOPHY
--------------------
All thresholds are FIXED and ABSOLUTE -- the same values apply to every file
across all sites, dates, and databases.

The one exception is the silent residual detector, which uses a within-
recording relative criterion (source RMS < 10% of the loudest sibling from
the same 8-source group). This is still absolute in spirit: a source that is
>20 dB quieter than its loudest sibling from the same original recording is
by definition a near-empty residual, regardless of site.

BAND-PASS AWARENESS
-------------------
Recordings in "bird_range/" have been band-pass filtered to
500-4000 Hz.  Recordings in "full_range/" retain the full frequency range.
Spectral entropy and spectral flatness are computed only over the active
frequency band so that a band-passed file is not penalised for having no
energy outside its pass-band.

SILENCE OVERRIDE (high-SNR isolated calls)
------------------------------------------
A file flagged as silent (low overall RMS) is rescued if it contains at least
one high-energy transient event -- i.e. a brief isolated call on a quiet
background, which is exactly the kind of well-separated close-range call we
want to keep.  The rescue criterion is:

    peak_frame_rms / median_frame_rms >= SNR_RESCUE_RATIO

where frame RMS is computed in short windows.  A ratio >= 10.0 (~20 dB above
the median frame) indicates a genuine acoustic event.

THREE INDEPENDENT DETECTORS (any one firing = noise)
-----------------------------------------------------
(A) SILENT RESIDUAL
      RMS < ABSOLUTE_RMS_FLOOR                    (catches truly silent files)
      OR RMS < 0.10 * max(RMS of all 8 siblings)  (catches relative residuals)
      UNLESS peak_frame_rms / median_frame_rms >= SNR_RESCUE_RATIO
             (rescues isolated high-SNR calls on quiet background)

(B) BROADBAND ENVIRONMENTAL NOISE  (wind, rain, stream)
      spectral_entropy  >= THR_SPECTRAL_ENTROPY     (flat spectrum)
      AND spectral_flatness >= THR_SPECTRAL_FLATNESS (Wiener entropy near 1)
      [computed over active frequency band only]

(C) TONAL INSECT BAND  (crickets, cicadas -- narrow persistent band)
      tpi              >= THR_TPI                   (one narrow band dominates)
      AND temporal_entropy >= THR_TEMPORAL_ENTROPY  (energy sustained over time)

FIXED THRESHOLD VALUES
----------------------
  THR_SPECTRAL_ENTROPY   = 0.85
    Sueur et al. (2008): random noise approaches Hf = 1.0; biologically
    active recordings typically 0.7-0.9.  0.85 is more aggressive than
    0.95 to catch low-level broadband noise.

  THR_SPECTRAL_FLATNESS  = 0.50
    Pure white noise = 1.0; structured signals typically < 0.3.
    Requiring BOTH Hf >= 0.85 AND SF >= 0.50 avoids false positives.

  THR_TPI               = 0.88
    Single narrow band dominates > 80% of frames -> insect drone.

  THR_TEMPORAL_ENTROPY   = 0.88
    Sustained drone spreads energy uniformly through time -> Ht near 1.0.
    Bird calls produce discrete pulses -> Ht typically 0.7-0.9.

  ABSOLUTE_RMS_FLOOR     = 1e-4
    16-bit audio noise floor ~3e-5; below 1e-4 = no real signal.

  WITHIN_RECORDING_RMS_RATIO = 0.10
    Source > 20 dB below loudest sibling = MixIT over-separation residual.

  SNR_RESCUE_RATIO       = 5.0
    peak_frame_rms / median_frame_rms >= 5.0 indicates a genuine
    isolated call on a quiet background; overrides silence detection.

Dependencies
------------
    pip install numpy torch soundfile librosa filelock
"""

import argparse
import csv
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from filelock import SoftFileLock

# ---------------------------------------------------------------------------
# Optional GPU backend
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

if not SOUNDFILE_AVAILABLE and not LIBROSA_AVAILABLE:
    print("ERROR: install at least one of: soundfile  OR  librosa")
    sys.exit(1)

DEVICE = None  # set in main()

# ---------------------------------------------------------------------------
# FIXED ABSOLUTE THRESHOLDS
# ---------------------------------------------------------------------------
THR_SPECTRAL_ENTROPY        = 0.85   # Hf >= this -> broadband noise
THR_SPECTRAL_FLATNESS       = 0.50   # SF >= this (with Hf) -> broadband noise
THR_TPI                     = 0.88   # TPI >= this -> possible insect band
THR_TEMPORAL_ENTROPY        = 0.88   # Ht >= this (with TPI) -> tonal insect
ABSOLUTE_RMS_FLOOR          = 1e-4   # RMS below this -> silent residual
WITHIN_RECORDING_RMS_RATIO  = 0.10   # RMS < ratio * group_max -> residual
SNR_RESCUE_RATIO            = 5.0   # peak/median frame RMS to rescue silence

# ---------------------------------------------------------------------------
# Band limits by top-level directory
# ---------------------------------------------------------------------------
BAND_LIMITS = {
    "bird_range": (500,    4000),
    "full_range": (0,     22050),
}
DEFAULT_BAND = (0, 22050)

# ---------------------------------------------------------------------------
# Output CSV path (single shared file, all tasks append to it)
# ---------------------------------------------------------------------------
MERGED_CSV = "[path to .csv keeping track of which files were classed as which and why]"

CSV_FIELDNAMES = [
    "file", "separated_dir", "top_dir", "label", "detector",
    "rms", "peak_median_snr",
    "spectral_entropy", "temporal_entropy", "spectral_flatness", "tpi",
    "band_hz_lo", "band_hz_hi",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(wav_path: Path):
    """Returns (y float32, sr) or (None, None)."""
    if SOUNDFILE_AVAILABLE:
        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            return y.astype(np.float32), int(sr)
        except Exception:
            pass
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            return y.astype(np.float32), int(sr)
        except Exception:
            pass
    return None, None


# ---------------------------------------------------------------------------
# STFT power (GPU-accelerated when available)
# ---------------------------------------------------------------------------

def stft_power(y: np.ndarray, sr: int,
               n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    """Returns power spectrogram (n_bins x n_frames)."""
    if TORCH_AVAILABLE and DEVICE is not None:
        t = torch.from_numpy(y).to(DEVICE)
        window = torch.hann_window(n_fft, device=DEVICE)
        S = torch.stft(t, n_fft=n_fft, hop_length=hop_length,
                       win_length=n_fft, window=window, return_complex=True)
        return S.abs().pow(2).cpu().numpy()
    else:
        return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2


def freq_bins(sr: int, n_fft: int, hz_lo: float, hz_hi: float):
    """Return (bin_lo, bin_hi) index slice for the given Hz range."""
    freqs  = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bin_lo = int(np.searchsorted(freqs, hz_lo))
    bin_hi = int(np.searchsorted(freqs, hz_hi, side="right"))
    bin_hi = min(bin_hi, len(freqs) - 1)
    return bin_lo, bin_hi


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)))


def peak_median_snr(y: np.ndarray, frame_len: int = 512) -> float:
    """
    Ratio of the 95th-percentile frame RMS to the median frame RMS.
    High value (>= SNR_RESCUE_RATIO) indicates an isolated high-energy
    transient event (close-range bird call) on a quiet background.
    Returns 0.0 if the signal is effectively silent throughout.
    """
    n_frames = len(y) // frame_len
    if n_frames < 4:
        return 0.0
    frames  = y[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per = np.sqrt((frames ** 2).mean(axis=1))
    median  = np.median(rms_per)
    if median < 1e-9:
        return 0.0
    peak = np.percentile(rms_per, 95)
    return float(peak / median)


def spectral_entropy_banded(S: np.ndarray, bin_lo: int, bin_hi: int) -> float:
    """
    Hf computed over frequency bins [bin_lo:bin_hi] only.
    Sueur et al. (2008). Range [0, 1]. Near 1 = flat/broadband.
    """
    S_band   = S[bin_lo:bin_hi, :]
    mean_spec = S_band.mean(axis=1)
    total    = mean_spec.sum()
    if total < 1e-12:
        return 1.0   # silent band -> treat as noise
    mean_spec /= total
    n = len(mean_spec)
    if n < 2:
        return 0.0
    hf = -np.sum(mean_spec * np.log(mean_spec + 1e-12))
    hf /= np.log(n)
    return float(np.clip(hf, 0.0, 1.0))


def spectral_flatness_banded(S: np.ndarray, bin_lo: int, bin_hi: int) -> float:
    """
    Wiener entropy over frequency bins [bin_lo:bin_hi] only.
    0 = pure tone, 1 = white noise.
    """
    S_band = S[bin_lo:bin_hi, :] + 1e-12
    log_mean = np.log(S_band).mean(axis=0)
    mean_log = np.log(S_band.mean(axis=0))
    return float(np.exp(log_mean - mean_log).mean())


def temporal_entropy(y: np.ndarray, sr: int, frame_len: int = 512) -> float:
    """Ht -- Sueur et al. (2008). Range [0, 1]. Near 1 = sustained drone."""
    if TORCH_AVAILABLE and DEVICE is not None:
        t = torch.from_numpy(y).to(DEVICE)
        n = len(t)
        Y = torch.fft.rfft(t)
        h = torch.zeros(len(Y), device=DEVICE)
        h[0] = 1
        h[1:(n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        envelope = torch.abs(torch.fft.irfft(Y * h, n=n)).cpu().numpy()
    else:
        envelope = np.abs(librosa.effects.harmonic(y))

    n_frames = len(envelope) // frame_len
    if n_frames < 2:
        return 1.0
    frames = envelope[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = frames.mean(axis=1) ** 2
    energy /= energy.sum() + 1e-12
    ht = -np.sum(energy * np.log(energy + 1e-12))
    ht /= np.log(len(energy))
    return float(np.clip(ht, 0.0, 1.0))


def tonal_persistence_index(S: np.ndarray, sr: int, n_fft: int,
                             peak_bandwidth_hz: float = 200.0,
                             dominance_threshold: float = 0.35) -> float:
    """
    TPI over the full spectrogram (insect bands can be anywhere).
    Vectorised -- no Python loop over frames.
    """
    hz_per_bin = (sr / 2.0) / (n_fft // 2)
    half_width = max(1, int(peak_bandwidth_hz / (2.0 * hz_per_bin)))

    total_energy = S.sum(axis=0)
    valid = total_energy > 1e-12
    if not valid.any():
        return 0.0

    peak_bins      = S.argmax(axis=0)
    n_bins, n_frames = S.shape
    lo = np.clip(peak_bins - half_width, 0, n_bins - 1)
    hi = np.clip(peak_bins + half_width + 1, 0, n_bins)

    cumS        = np.concatenate([np.zeros((1, n_frames)), S.cumsum(axis=0)], axis=0)
    idx         = np.arange(n_frames)
    peak_energy = cumS[hi, idx] - cumS[lo, idx]

    dominant = valid & (peak_energy / (total_energy + 1e-12) >= dominance_threshold)
    return float(dominant.sum() / n_frames)


# ---------------------------------------------------------------------------
# Per-file feature extraction
# ---------------------------------------------------------------------------

def extract_features(wav_path: Path,
                     hz_lo: float, hz_hi: float) -> dict | None:
    """
    Extract all features.  Spectral entropy and flatness are computed only
    over [hz_lo, hz_hi] to be fair to band-passed recordings.
    """
    y, sr = load_audio(wav_path)
    if y is None or len(y) == 0 or not np.isfinite(y).all():
        return None

    n_fft      = 1024
    hop_length = 512
    S          = stft_power(y, sr, n_fft=n_fft, hop_length=hop_length)
    blo, bhi   = freq_bins(sr, n_fft, hz_lo, hz_hi)

    return {
        "path":             str(wav_path),
        "sr":               sr,
        "duration_s":       len(y) / sr,
        "rms":              rms_energy(y),
        "peak_median_snr":  peak_median_snr(y),
        "spectral_entropy": spectral_entropy_banded(S, blo, bhi),
        "temporal_entropy": temporal_entropy(y, sr),
        "spectral_flatness":spectral_flatness_banded(S, blo, bhi),
        "tpi":              tonal_persistence_index(S, sr, n_fft),
        "band_hz_lo":       hz_lo,
        "band_hz_hi":       hz_hi,
    }


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

RECORDING_RE = re.compile(r"^(.+_\d{8}_\d{6})_(source\d+)\.wav$")


def parse_filename(wav_path: Path):
    m = RECORDING_RE.match(wav_path.name)
    return (m.group(1), m.group(2)) if m else (None, None)


# ---------------------------------------------------------------------------
# Three independent detectors
# ---------------------------------------------------------------------------

def classify(feats: dict, group_rms_max: float | None) -> tuple:
    """Returns (label, detector_name_or_None)."""

    snr = feats["peak_median_snr"]

    # Catch extreme entropy before the paired logic
    if feats["temporal_entropy"] >= 0.95:
        return "noise", "extreme_temporal_entropy"

    if feats["spectral_entropy"] >= 0.90:
        return "noise", "extreme_spectral_entropy"

    # (A) Silent residual -- absolute floor
    if feats["rms"] < ABSOLUTE_RMS_FLOOR:
        if snr >= SNR_RESCUE_RATIO:
            pass   # rescued: isolated call on near-silent background
        else:
            return "noise", "silent_residual_absolute"

    # (A) Silent residual -- within-recording relative
    if group_rms_max is not None and group_rms_max > 0:
        if feats["rms"] < WITHIN_RECORDING_RMS_RATIO * group_rms_max:
            if snr >= SNR_RESCUE_RATIO:
                pass   # rescued
            else:
                return "noise", "silent_residual_relative"

    # (B) Broadband environmental noise
    if (feats["spectral_entropy"]  >= THR_SPECTRAL_ENTROPY and
            feats["spectral_flatness"] >= THR_SPECTRAL_FLATNESS):
        return "noise", "broadband_noise"

    # (C) Tonal insect band
    if (feats["tpi"]                  >= THR_TPI and
            feats["temporal_entropy"] >= THR_TEMPORAL_ENTROPY):
        return "noise", "tonal_insect"

    return "active", None


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

#TOP_DIRS = ["postsplit", "bird_range", "full_range"]
#DB_DIRS  = ["PER", "STM_hard", "STM_soft", "STM_s_LTR"]
TOP_DIRS = ["bird_range", "full_range"]
DB_DIRS  = ["PER", "STM_h", "STM_s_HTR", "STM_s_LTR"]


def find_separated_dirs(root: Path) -> list:
    """Returns list of (sep_dir, top_dir_name) tuples."""
    found = []
    for top in TOP_DIRS:
        for db in DB_DIRS:
            sep = root / top / db / "separated"
            if sep.is_dir():
                log.info("Found: %s", sep)
                found.append((sep, top))
            else:
                log.warning("Expected dir NOT FOUND: %s", sep)
    return found


# ---------------------------------------------------------------------------
# Merged CSV writer (file-locked for concurrent tasks)
# ---------------------------------------------------------------------------

def append_rows_to_csv_safe(rows: list, csv_path: str):
    """
    Appends rows to the shared CSV using filelock. 
    A physical lock file guarantees no overwrites on HPC network drives.
    """
    if not rows:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # This creates a physical lock file (classification_results.csv.lock)
    # The timeout ensures tasks wait patiently if another task is writing
    lock = SoftFileLock(str(csv_path) + ".lock", timeout=120)

    with lock:
        write_header = not path.exists() or path.stat().st_size == 0
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Process one separated/ directory
# ---------------------------------------------------------------------------

def process_dir(sep_dir: Path, top_dir: str, my_files: list,
                dry_run: bool, rows: list):
    hz_lo, hz_hi = BAND_LIMITS.get(top_dir, DEFAULT_BAND)
    log.info("--- %s  (%d files)  band=%d-%d Hz",
             sep_dir, len(my_files), hz_lo, hz_hi)

    noise_dir     = sep_dir.parent / "noise"
    active_dir = sep_dir.parent / "active"
    if not dry_run:
        noise_dir.mkdir(exist_ok=True)
        active_dir.mkdir(exist_ok=True)

    # Group by recording ID
    by_recording = defaultdict(list)
    ungrouped    = []
    for wf in my_files:
        rid, _ = parse_filename(wf)
        if rid:
            by_recording[rid].append(wf)
        else:
            ungrouped.append(wf)

    log.info("  %d recordings, %d ungrouped", len(by_recording), len(ungrouped))
    done = 0

    for rid, group in by_recording.items():
        group_feats = []
        for wf in sorted(group):
            f = extract_features(wf, hz_lo, hz_hi)
            if f:
                group_feats.append(f)
            else:
                log.warning("    Skipping unreadable: %s", wf.name)

        if not group_feats:
            continue

        group_rms_max = max(f["rms"] for f in group_feats)

        for feats in group_feats:
            label, detector = classify(feats, group_rms_max)
            _copy_and_record(feats, label, detector, top_dir,
                             noise_dir, active_dir, dry_run, rows)

        append_rows_to_csv_safe(rows, MERGED_CSV)
        rows.clear()

        done += len(group_feats)
        if done % 1000 == 0:
            log.info("  %d / %d done ...", done, len(my_files))

    for wf in ungrouped:
        f = extract_features(wf, hz_lo, hz_hi)
        if not f:
            log.warning("    Skipping unreadable: %s", wf.name)
            continue
        label, detector = classify(f, group_rms_max=None)
        _copy_and_record(f, label, detector, top_dir,
                         noise_dir, active_dir, dry_run, rows)

        append_rows_to_csv_safe(rows, MERGED_CSV)
        rows.clear()


    log.info("  Finished: %d files", done + len(ungrouped))


def _copy_and_record(feats, label, detector, top_dir,
                     noise_dir, active_dir, dry_run, rows):
    src = Path(feats["path"])
    dst = (noise_dir if label == "noise" else active_dir) / src.name

    tag = f"NOISE [{detector}]" if label == "noise" else "active          "
    log.info(
        "  %s  %s  rms=%.5f snr=%5.1f Hf=%.3f Ht=%.3f SF=%.3f TPI=%.3f",
        tag, src.name,
        feats["rms"], feats["peak_median_snr"],
        feats["spectral_entropy"], feats["temporal_entropy"],
        feats["spectral_flatness"], feats["tpi"],
    )

    if not dry_run:
        shutil.copy2(str(src), str(dst))

    rows.append({
        "file":             src.name,
        "separated_dir":    str(src.parent),
        "top_dir":          top_dir,
        "label":            label,
        "detector":         detector or "",
        "rms":              feats["rms"],
        "peak_median_snr":  feats["peak_median_snr"],
        "spectral_entropy": feats["spectral_entropy"],
        "temporal_entropy": feats["temporal_entropy"],
        "spectral_flatness":feats["spectral_flatness"],
        "tpi":              feats["tpi"],
        "band_hz_lo":       feats["band_hz_lo"],
        "band_hz_hi":       feats["band_hz_hi"],
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_ROOT = "[path to root]"


def main():
    global DEVICE

    ap = argparse.ArgumentParser()
    ap.add_argument("--root",      default=DEFAULT_ROOT)
    ap.add_argument("--task_id",   type=int, default=0)
    ap.add_argument("--num_tasks", type=int, default=1)
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--no-gpu",    action="store_true")
    args = ap.parse_args()

    # Device
    if TORCH_AVAILABLE and not args.no_gpu and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu") if TORCH_AVAILABLE else None
        log.info("CPU mode (%s)", "torch" if TORCH_AVAILABLE else "librosa")

    log.info(
        "Thresholds: Hf>=%.2f  SF>=%.2f  TPI>=%.2f  Ht>=%.2f  "
        "RMS_floor=%.0e  RMS_ratio=%.2f  SNR_rescue=%.1f",
        THR_SPECTRAL_ENTROPY, THR_SPECTRAL_FLATNESS,
        THR_TPI, THR_TEMPORAL_ENTROPY,
        ABSOLUTE_RMS_FLOOR, WITHIN_RECORDING_RMS_RATIO, SNR_RESCUE_RATIO,
    )

    root = Path(args.root)
    if not root.is_dir():
        log.error("Root not found: %s", root)
        sys.exit(1)

    sep_dirs = sorted(find_separated_dirs(root), key=lambda x: str(x[0]))
    if not sep_dirs:
        log.error("No separated/ dirs found under %s", root)
        sys.exit(1)

    log.info("Task %d / %d  |  %d separated/ dirs",
             args.task_id, args.num_tasks, len(sep_dirs))

    rows = []

    for sep_dir, top_dir in sep_dirs:
        all_wav = sorted(sep_dir.glob("*.wav"))
        if not all_wav:
            log.warning("No .wav files in %s", sep_dir)
            continue

        # Partition by recording group across tasks
        groups     = defaultdict(list)
        ungrouped  = []
        for wf in all_wav:
            rid, _ = parse_filename(wf)
            if rid:
                groups[rid].append(wf)
            else:
                ungrouped.append(wf)

        sorted_rids = sorted(groups.keys())
        my_rids     = sorted_rids[args.task_id :: args.num_tasks]
        my_files    = [wf for rid in my_rids for wf in groups[rid]]
        my_files   += ungrouped[args.task_id :: args.num_tasks]

        log.info("%s (%s): %d total files, %d groups | task gets %d files",
                 sep_dir.name, top_dir, len(all_wav),
                 len(sorted_rids), len(my_files))

        if my_files:
            process_dir(sep_dir, top_dir, my_files, args.dry_run, rows)

    # Append this task's rows to the shared merged CSV
    log.info("Appended %d rows -> %s", len(rows), MERGED_CSV)

    if rows:
        total   = len(rows)
        n_noise = sum(1 for r in rows if r["label"] == "noise")
        by_det: dict = {}
        for r in rows:
            d = r["detector"]
            if d:
                by_det[d] = by_det.get(d, 0) + 1
        log.info("=== Task %d summary ===", args.task_id)
        log.info("Processed : %d", total)
        log.info("noise     : %d  (%.1f%%)", n_noise, 100 * n_noise / total)
        for d, n in sorted(by_det.items()):
            log.info("  %-35s : %d", d, n)
        log.info("active : %d  (%.1f%%)", total - n_noise,
                 100 * (total - n_noise) / total)
    else:
        log.warning("No files processed by task %d.", args.task_id)


if __name__ == "__main__":
    main()
