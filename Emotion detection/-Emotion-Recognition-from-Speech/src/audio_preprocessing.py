"""Deterministic audio preprocessing.

The exact same function is used for training, validation, test and inference.
That is the only way the numbers a deployed model produces mean anything.

Pipeline:  load mono @16 kHz -> trim silence -> peak-normalise -> fix length
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def load_audio(path: str | Path, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at the target sampling rate."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = C.TRIM_TOP_DB) -> np.ndarray:
    """Remove leading/trailing silence only.

    RAVDESS clips contain roughly half a second of room tone before and after
    the sentence.  That silence carries no emotional information but does eat
    into our fixed 3 s window, so we strip the edges.  Internal pauses are
    kept: pause structure is itself an emotional cue (sad speech pauses more).
    """
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Scale to unit peak so absolute recording gain cannot be used as a cue."""
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def fix_length(y: np.ndarray, n_samples: int = C.N_SAMPLES) -> np.ndarray:
    """Centre-crop or symmetrically zero-pad to an exact number of samples.

    Centre alignment keeps the utterance nucleus (where the emotional prosody
    peaks) inside the window for both short and long clips.
    """
    if len(y) == n_samples:
        return y
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start:start + n_samples]
    pad = n_samples - len(y)
    left = pad // 2
    return np.pad(y, (left, pad - left), mode="constant")


def preprocess(path: str | Path) -> np.ndarray:
    """Full deterministic preprocessing chain for a single file."""
    y = load_audio(path)
    y = trim_silence(y)
    if C.PEAK_NORM:
        y = peak_normalize(y)
    return fix_length(y)


def preprocess_waveform(y: np.ndarray) -> np.ndarray:
    """Same chain applied to an already-loaded waveform (used by augmentation)."""
    y = trim_silence(y)
    if C.PEAK_NORM:
        y = peak_normalize(y)
    return fix_length(y)
