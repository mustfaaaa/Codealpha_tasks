"""Waveform augmentation -- TRAINING DATA ONLY.

Rationale: RAVDESS gives us ~1000 training clips from 16 actors, which is very
small for a convolutional network.  Augmentation multiplies the effective
sample count and, more importantly, forces invariance to nuisance factors
(recording noise, speaking rate, absolute pitch, microphone gain) that are not
emotion.

Constraint: every transform must preserve the emotion label.  Ranges are kept
deliberately mild -- a large pitch shift or an aggressive stretch would start
to *change* the perceived affect (speeding up neutral speech makes it sound
agitated), which would corrupt the labels we are training on.

Validation and test data are NEVER augmented: their job is to estimate
performance on real, untouched audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def add_noise(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at a controlled SNR."""
    snr_db = rng.uniform(*C.AUG_NOISE_SNR_DB)
    sig_power = np.mean(y ** 2)
    if sig_power <= 0:
        return y
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def time_stretch(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Change speaking rate without changing pitch."""
    rate = rng.uniform(*C.AUG_TIME_STRETCH)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def pitch_shift(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift pitch by at most two semitones (keeps the voice plausible)."""
    steps = rng.uniform(*C.AUG_PITCH_STEPS)
    return librosa.effects.pitch_shift(
        y, sr=C.SAMPLE_RATE, n_steps=steps).astype(np.float32)


def gain(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random level change (removed again by peak normalisation, so applied
    before noise in the chain below to alter the effective SNR)."""
    db = rng.uniform(*C.AUG_GAIN_DB)
    return (y * (10 ** (db / 20))).astype(np.float32)


TRANSFORMS = [add_noise, time_stretch, pitch_shift, gain]


def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply 1-2 randomly chosen transforms to a raw waveform."""
    k = rng.integers(1, 3)
    chosen = rng.choice(len(TRANSFORMS), size=int(k), replace=False)
    out = y.copy()
    for i in chosen:
        out = TRANSFORMS[int(i)](out, rng)
    return out
