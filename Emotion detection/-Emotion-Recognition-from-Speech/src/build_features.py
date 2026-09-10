"""Offline feature extraction into .npz caches.

Features are computed once and reused by every model so that the baseline,
the CNN, the LSTM and the CNN-LSTM all consume exactly the same audio.

Two training sets are produced:
  * train_clean  -- one feature tensor per original file
  * train_aug    -- train_clean plus AUG_COPIES augmented variants per file

Validation and test are extracted clean only, and never augmented.
Standardisation statistics are computed on the TRAINING split only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from augmentation import augment  # noqa: E402
from audio_preprocessing import load_audio, preprocess_waveform  # noqa: E402
from feature_extraction import features_from_waveform  # noqa: E402


def extract_split(df: pd.DataFrame, augment_copies: int = 0,
                  seed: int = C.SEED, tag: str = "") -> dict[str, np.ndarray]:
    """Extract all three feature views for every row, optionally augmented."""
    rng = np.random.default_rng(seed)
    cnn, seq, stats, labels, actors, files, is_aug = [], [], [], [], [], [], []
    t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        raw = load_audio(r["path"])
        variants = [(preprocess_waveform(raw), 0)]
        for _ in range(augment_copies):
            variants.append((preprocess_waveform(augment(raw, rng)), 1))
        for wav, flag in variants:
            f = features_from_waveform(wav)
            cnn.append(f["cnn"])
            seq.append(f["seq"])
            stats.append(f["stats"])
            labels.append(r["emotion"])
            actors.append(int(r["actor"]))
            files.append(r["filename"])
            is_aug.append(flag)
        if (i + 1) % 200 == 0:
            print(f"  [{tag}] {i + 1}/{len(df)} files  ({time.time() - t0:.0f}s)")
    return {
        "cnn": np.asarray(cnn, dtype=np.float32),
        "seq": np.asarray(seq, dtype=np.float32),
        "stats": np.asarray(stats, dtype=np.float32),
        "label": np.asarray(labels),
        "actor": np.asarray(actors),
        "filename": np.asarray(files),
        "is_aug": np.asarray(is_aug),
    }


def main() -> None:
    manifest = pd.read_csv(C.DATA_PROCESSED / "manifest.csv")
    manifest = manifest[manifest["error"].fillna("") == ""].reset_index(drop=True)
    # RAVDESS ships one byte-identical duplicate pair (actor 07, happy,
    # statement 02, repetitions 01/02).  Both land in train, so this is not a
    # leak, but keeping both would double-weight that utterance.
    before = len(manifest)
    manifest = manifest.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    if before != len(manifest):
        print(f"dropped {before - len(manifest)} byte-duplicate file(s)")

    tr = manifest[manifest.split == "train"].reset_index(drop=True)
    va = manifest[manifest.split == "val"].reset_index(drop=True)
    te = manifest[manifest.split == "test"].reset_index(drop=True)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    out = C.DATA_PROCESSED
    packs = {
        "train_clean": extract_split(tr, 0, C.SEED, "train_clean"),
        "train_aug": extract_split(tr, C.AUG_COPIES, C.SEED + 1, "train_aug"),
        "val": extract_split(va, 0, C.SEED, "val"),
        "test": extract_split(te, 0, C.SEED, "test"),
    }
    for name, p in packs.items():
        np.savez_compressed(out / f"features_{name}.npz", **p)
        print(f"{name}: cnn={p['cnn'].shape} seq={p['seq'].shape} "
              f"stats={p['stats'].shape}")

    # Standardisation statistics -- TRAIN ONLY (clean, un-augmented).
    tc = packs["train_clean"]
    norm = {
        "cnn_mean": tc["cnn"].mean(axis=(0, 1, 2)).tolist(),
        "cnn_std": (tc["cnn"].std(axis=(0, 1, 2)) + 1e-8).tolist(),
        "seq_mean": tc["seq"].mean(axis=(0, 1)).tolist(),
        "seq_std": float(tc["seq"].std() + 1e-8),
        "stats_mean": tc["stats"].mean(axis=0).tolist(),
        "stats_std": (tc["stats"].std(axis=0) + 1e-8).tolist(),
    }
    (out / "norm_stats.json").write_text(json.dumps(norm))

    cfg = {
        "sample_rate": C.SAMPLE_RATE, "duration_sec": C.DURATION,
        "n_samples": C.N_SAMPLES, "trim_top_db": C.TRIM_TOP_DB,
        "peak_normalize": C.PEAK_NORM, "n_mfcc": C.N_MFCC, "n_fft": C.N_FFT,
        "hop_length": C.HOP_LENGTH, "win_length": C.WIN_LENGTH,
        "n_mels": C.N_MELS, "fmin": C.FMIN, "fmax": C.FMAX,
        "n_frames": C.N_FRAMES, "cnn_input_shape": list(C.CNN_INPUT_SHAPE),
        "seq_input_shape": list(C.SEQ_INPUT_SHAPE),
        "channels": ["mfcc", "delta", "delta2"],
        "aug_copies_train": C.AUG_COPIES, "seed": C.SEED,
    }
    (out / "feature_config.json").write_text(json.dumps(cfg, indent=2))
    print("saved norm_stats.json and feature_config.json")


if __name__ == "__main__":
    main()
