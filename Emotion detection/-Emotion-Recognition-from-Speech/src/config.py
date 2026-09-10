"""Central configuration for the Speech Emotion Recognition project.

Every constant that influences preprocessing, feature extraction or the
train/val/test partition lives here so that training and inference are
guaranteed to use identical settings.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RAVDESS_DIR = DATA_RAW / "RAVDESS"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
METRICS_DIR = OUTPUTS / "metrics"
PRED_DIR = OUTPUTS / "predictions"

for _p in (DATA_PROCESSED, MODELS_DIR, FIG_DIR, METRICS_DIR, PRED_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# RAVDESS label maps (from the official filename identifier specification)
# 7 fields: modality-vocalChannel-emotion-intensity-statement-repetition-actor
# ----------------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

# Deterministic, gender-balanced, ACTOR-DISJOINT partition.
# Odd actor id = male, even actor id = female (RAVDESS convention).
TEST_ACTORS = [20, 21, 22, 23]   # 2 male (21, 23) + 2 female (20, 22)
VAL_ACTORS = [16, 17, 18, 19]    # 2 male (17, 19) + 2 female (16, 18)
TRAIN_ACTORS = [a for a in range(1, 25) if a not in TEST_ACTORS + VAL_ACTORS]

# ----------------------------------------------------------------------------
# Audio preprocessing  (identical for train / val / test / inference)
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000       # 16 kHz keeps all speech-relevant energy (<8 kHz)
DURATION = 3.0             # seconds, fixed-length input
N_SAMPLES = int(SAMPLE_RATE * DURATION)
TRIM_TOP_DB = 30           # silence trimming threshold (RAVDESS has lead-in silence)
PEAK_NORM = True           # peak-normalise amplitude to remove recording-gain bias

# ----------------------------------------------------------------------------
# MFCC / spectral feature parameters
# ----------------------------------------------------------------------------
N_MFCC = 40
N_FFT = 2048               # 128 ms window at 16 kHz
HOP_LENGTH = 512           # 32 ms hop  -> 94 frames for a 3 s clip
WIN_LENGTH = 2048
N_MELS = 128
FMIN = 20
FMAX = 8000
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH   # 94

# Derived tensor shapes
CNN_INPUT_SHAPE = (N_MFCC, N_FRAMES, 3)          # (40, 94, 3) mfcc/delta/delta2
SEQ_INPUT_SHAPE = (N_FRAMES, N_MFCC * 3)         # (94, 120) per-frame vectors

# ----------------------------------------------------------------------------
# Augmentation (TRAIN ONLY)
# ----------------------------------------------------------------------------
AUG_COPIES = 2             # extra augmented copies per training clip
AUG_NOISE_SNR_DB = (15.0, 30.0)
AUG_TIME_STRETCH = (0.9, 1.1)
AUG_PITCH_STEPS = (-2.0, 2.0)
AUG_GAIN_DB = (-6.0, 6.0)

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
BATCH_SIZE = 32
# BatchNorm moving-average momentum.  Lowered from the Keras default of 0.99
# because this dataset gives very few steps per epoch (see models.build_cnn).
BN_MOMENTUM = 0.9
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 14
LR = 5e-4
