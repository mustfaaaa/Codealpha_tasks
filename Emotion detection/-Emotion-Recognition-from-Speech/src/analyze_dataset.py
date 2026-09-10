"""PHASE 3 -- dataset analysis, audio validation and exploratory figures.

Everything printed here is computed from the actual files on disk.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import assign_split, build_manifest  # noqa: E402
from audio_preprocessing import preprocess  # noqa: E402

sns.set_theme(style="whitegrid")
EMO_ORDER = list(C.EMOTION_MAP.values())


def validate_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Open every file: record duration, sample rate, channels, md5, errors."""
    rows = []
    for i, r in df.iterrows():
        rec = {"filename": r["filename"], "error": ""}
        try:
            info = sf.info(r["path"])
            rec.update(
                duration=info.duration, sr=info.samplerate,
                channels=info.channels, frames=info.frames, subtype=info.subtype,
            )
            with open(r["path"], "rb") as fh:
                rec["md5"] = hashlib.md5(fh.read()).hexdigest()
            if info.frames == 0:
                rec["error"] = "empty"
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        if (i + 1) % 300 == 0:
            print(f"  validated {i + 1}/{len(df)}")
    return pd.DataFrame(rows)


def main() -> None:
    df = assign_split(build_manifest())
    print(f"Indexed {len(df)} speech files")

    val = validate_audio(df)
    df = df.merge(val, on="filename", how="left")

    corrupt = df[df["error"] != ""]
    dup_md5 = df[df.duplicated("md5", keep=False)]

    counts = df["emotion"].value_counts()
    summary = {
        "total_recordings": int(len(df)),
        "total_speakers": int(df["actor"].nunique()),
        "total_emotions": int(df["emotion"].nunique()),
        "emotions": EMO_ORDER,
        "corrupt_or_unreadable": int(len(corrupt)),
        "duplicate_md5_files": int(len(dup_md5)),
        "duplicate_md5_filenames": sorted(dup_md5["filename"].tolist()),
        "stereo_files": sorted(df.loc[df["channels"] == 2, "filename"].tolist()),
        "missing_files": int(df["path"].map(lambda p: not Path(p).exists()).sum()),
        "sampling_rates": {str(k): int(v) for k, v in df["sr"].value_counts().items()},
        "channels": {str(k): int(v) for k, v in df["channels"].value_counts().items()},
        "duration_sec": {
            "min": float(df["duration"].min()),
            "max": float(df["duration"].max()),
            "mean": float(df["duration"].mean()),
            "median": float(df["duration"].median()),
            "std": float(df["duration"].std()),
        },
        "class_distribution": {k: int(v) for k, v in counts.reindex(EMO_ORDER).items()},
        "gender_distribution": {k: int(v) for k, v in df["gender"].value_counts().items()},
        "intensity_distribution": {k: int(v) for k, v in df["intensity"].value_counts().items()},
        "files_per_actor": {str(k): int(v) for k, v in df["actor"].value_counts().sort_index().items()},
        "split_files": {k: int(v) for k, v in df["split"].value_counts().items()},
        "split_actors": {s: sorted(int(a) for a in g["actor"].unique())
                         for s, g in df.groupby("split")},
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
    }

    (C.METRICS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(C.DATA_PROCESSED / "manifest.csv", index=False)

    print(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- figures
    # Q: is any emotion under-represented, i.e. do we need class weighting?
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.countplot(data=df, x="emotion", order=EMO_ORDER, hue="emotion",
                  palette="viridis", legend=False, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("RAVDESS speech: recordings per emotion")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "01_emotion_distribution.png", dpi=140)
    plt.close(fig)

    # Q: how long are the clips, i.e. what fixed window should we use?
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df["duration"], bins=40, kde=True, color="steelblue", ax=ax)
    ax.axvline(C.DURATION, color="crimson", ls="--",
               label=f"chosen window = {C.DURATION}s")
    ax.set_xlabel("raw duration (s)")
    ax.legend()
    ax.set_title("Clip duration distribution (raw, before silence trimming)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "02_duration_distribution.png", dpi=140)
    plt.close(fig)

    # Q: is the actor-disjoint split still class-balanced in each partition?
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.countplot(data=df, x="emotion", hue="split", order=EMO_ORDER,
                  hue_order=["train", "val", "test"], palette="Set2", ax=ax)
    ax.set_title("Class distribution within each speaker-independent split")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "03_split_distribution.png", dpi=140)
    plt.close(fig)

    # Q: does every actor contribute the same amount of data?
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cnt = df.groupby(["actor", "split"]).size().reset_index(name="n")
    sns.barplot(data=cnt, x="actor", y="n", hue="split",
                hue_order=["train", "val", "test"], palette="Set2",
                dodge=False, ax=ax)
    ax.set_title("Recordings per actor, coloured by partition (actors never shared)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "04_actor_distribution.png", dpi=140)
    plt.close(fig)

    # Q: what do waveform / mel / MFCC look like for contrasting emotions?
    picks = ["neutral", "happy", "angry", "sad"]
    fig, axes = plt.subplots(3, len(picks), figsize=(4.2 * len(picks), 9))
    for j, emo in enumerate(picks):
        row = df[(df.emotion == emo) & (df.split == "train")].iloc[0]
        y = preprocess(row["path"])
        librosa.display.waveshow(y, sr=C.SAMPLE_RATE, ax=axes[0, j])
        axes[0, j].set_title(f"{emo} - waveform")
        axes[0, j].set_xlabel("")
        M = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
            n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX), ref=np.max)
        librosa.display.specshow(M, sr=C.SAMPLE_RATE, hop_length=C.HOP_LENGTH,
                                 x_axis="time", y_axis="mel", fmax=C.FMAX,
                                 ax=axes[1, j])
        axes[1, j].set_title(f"{emo} - mel spectrogram (dB)")
        mf = librosa.feature.mfcc(y=y, sr=C.SAMPLE_RATE, n_mfcc=C.N_MFCC,
                                  n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                                  n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
        librosa.display.specshow(mf, sr=C.SAMPLE_RATE, hop_length=C.HOP_LENGTH,
                                 x_axis="time", ax=axes[2, j])
        axes[2, j].set_title(f"{emo} - MFCC ({C.N_MFCC})")
    fig.suptitle("Identical preprocessing chain, four emotions", y=1.001)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "05_waveform_mel_mfcc.png", dpi=130)
    plt.close(fig)

    # Q: do simple acoustic descriptors already separate the emotions?
    feats = {"rms_mean": [], "zcr_mean": [], "centroid_mean": [], "emotion": []}
    sub = df.groupby("emotion", group_keys=False).head(60)
    for _, r in sub.iterrows():
        y = preprocess(r["path"])
        feats["rms_mean"].append(float(librosa.feature.rms(y=y).mean()))
        feats["zcr_mean"].append(float(librosa.feature.zero_crossing_rate(y).mean()))
        feats["centroid_mean"].append(
            float(librosa.feature.spectral_centroid(y=y, sr=C.SAMPLE_RATE).mean()))
        feats["emotion"].append(r["emotion"])
    fd = pd.DataFrame(feats)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, lab in zip(axes,
                            ["rms_mean", "centroid_mean", "zcr_mean"],
                            ["RMS energy (loudness)",
                             "Spectral centroid (brightness)",
                             "Zero-crossing rate (noisiness)"]):
        sns.boxplot(data=fd, x="emotion", y=col, order=EMO_ORDER,
                    hue="emotion", palette="viridis", legend=False, ax=ax)
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
    fig.suptitle("Low-level descriptors by emotion (60 clips per class)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "06_feature_comparison.png", dpi=140)
    plt.close(fig)
    fd.to_csv(C.METRICS_DIR / "lowlevel_feature_probe.csv", index=False)

    print("figures written to", C.FIG_DIR)


if __name__ == "__main__":
    main()
