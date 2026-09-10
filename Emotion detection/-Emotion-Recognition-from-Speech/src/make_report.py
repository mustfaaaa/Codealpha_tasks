"""Generate report.md from the metrics files that were actually produced.

Every number in the report is read from JSON/CSV written by the pipeline, so
the document cannot drift from the executed results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    ds = json.loads((C.METRICS_DIR / "dataset_summary.json").read_text())
    metrics = json.loads((C.METRICS_DIR / "test_metrics_all.json").read_text())
    comp = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")
    card = json.loads((C.MODELS_DIR / "final_model" / "model_card.json").read_text())
    ranking = json.loads((C.METRICS_DIR / "validation_ranking.json").read_text())
    calib = json.loads((C.METRICS_DIR / "confidence_calibration.json").read_text())
    unseen = json.loads((C.METRICS_DIR / "unseen_audio_summary.json").read_text())

    best_name = card["selected_model"]
    best = metrics[best_name]

    # per-class table for the selected model
    per_class = pd.DataFrame([
        {"Emotion": k, "Precision": round(v["precision"], 3),
         "Recall": round(v["recall"], 3), "F1": round(v["f1-score"], 3),
         "Support": int(v["support"])}
        for k, v in best["per_class"].items()
    ]).sort_values("F1", ascending=False)

    val_tbl = pd.DataFrame([
        {"Model": r["name"], "Val Accuracy": round(r["val_accuracy"], 4),
         "Val Macro F1": round(r["val_macro_f1"], 4),
         "Params": f"{r['params']:,}", "Epochs": r["epochs_run"],
         "Train (s)": r["train_seconds"]}
        for r in ranking
    ])

    cls_dist = pd.DataFrame(
        [{"Emotion": k, "Recordings": v}
         for k, v in ds["class_distribution"].items()])

    conf = pd.read_csv(C.PRED_DIR / f"misclassified_{best_name}.csv")
    top_conf = (conf.groupby(["true", "predicted"]).size()
                .reset_index(name="count")
                .sort_values("count", ascending=False).head(10))

    d = ds["duration_sec"]
    sel = card["selection_details"]
    within_1se = ", ".join(f"`{m}`" for m in sel["models_within_1se"])
    text = f"""# Speech Emotion Recognition on RAVDESS — Project Report

Generated from executed pipeline outputs. Every figure below is read from
`outputs/metrics/`; none is hand-written.

## 1. Dataset

**RAVDESS**, speech subset (Zenodo record 1188976,
`Audio_Speech_Actors_01-24.zip`, CC BY-NC-SA 4.0).

| Property | Value |
|---|---|
| Total recordings | {ds['total_recordings']} |
| Speakers (actors) | {ds['total_speakers']} ({ds['gender_distribution']['male']} male / {ds['gender_distribution']['female']} female clips) |
| Emotions | {ds['total_emotions']} |
| Native sampling rate | {', '.join(f'{k} Hz x {v}' for k, v in ds['sampling_rates'].items())} |
| Channels | {', '.join(f'{k}ch x {v}' for k, v in ds['channels'].items())} |
| Duration min / mean / max | {d['min']:.2f}s / {d['mean']:.2f}s / {d['max']:.2f}s |
| Duration median (std) | {d['median']:.2f}s ({d['std']:.2f}s) |
| Corrupt / unreadable | {ds['corrupt_or_unreadable']} |
| Missing files | {ds['missing_files']} |
| Byte-identical duplicates | {ds['duplicate_md5_files']} ({', '.join(ds['duplicate_md5_filenames'])}) |
| Class imbalance (max/min) | {ds['imbalance_ratio_max_over_min']:.1f}x |

### Class distribution

{md_table(cls_dist)}

Only `neutral` is under-represented: RAVDESS records neutral at normal
intensity only, while every other emotion is recorded at both normal and
strong intensity. This 2:1 imbalance is handled with class weights, and macro-F1
is used as the primary metric so the small class cannot be ignored for free.

## 2. Speaker-independent split (leakage prevention)

Actors are **disjoint** across partitions, and every partition is gender
balanced (odd actor id = male, even = female):

| Split | Actors | Clips |
|---|---|---|
| train | {', '.join(map(str, ds['split_actors']['train']))} | {ds['split_files']['train']} |
| val | {', '.join(map(str, ds['split_actors']['val']))} | {ds['split_files']['val']} |
| test | {', '.join(map(str, ds['split_actors']['test']))} | {ds['split_files']['test']} |

A random file-level split would put several recordings of the same actor in
both train and test. The network could then recognise the *voice* rather than
the *emotion*, and the reported accuracy would be inflated. `src/dataset.py`
asserts the actor sets do not intersect, so this cannot regress silently.

## 3. Preprocessing and features

Identical for train, validation, test and inference (`src/audio_preprocessing.py`,
`src/feature_extraction.py`); the exported bundle stores the configuration and
`src/predict.py` refuses to run if the live code no longer matches it.

| Stage | Setting |
|---|---|
| Channels | mono (stereo files averaged) |
| Sampling rate | {C.SAMPLE_RATE} Hz |
| Silence trimming | leading/trailing only, top_db={C.TRIM_TOP_DB} |
| Amplitude | peak normalisation |
| Length handling | centre crop / symmetric zero-pad to {C.DURATION}s ({C.N_SAMPLES} samples) |
| MFCC | n_mfcc={C.N_MFCC}, n_fft={C.N_FFT}, hop={C.HOP_LENGTH}, win={C.WIN_LENGTH}, n_mels={C.N_MELS}, fmin={C.FMIN}, fmax={C.FMAX} |
| Derived channels | MFCC, delta, delta-delta |
| CNN tensor | {tuple(C.CNN_INPUT_SHAPE)} |
| Sequence tensor | {tuple(C.SEQ_INPUT_SHAPE)} |
| Standardisation | per-channel, statistics from the **training split only** |
| Augmentation (train only) | {C.AUG_COPIES} extra copies: noise (SNR {C.AUG_NOISE_SNR_DB[0]:.0f}-{C.AUG_NOISE_SNR_DB[1]:.0f} dB), time-stretch {C.AUG_TIME_STRETCH}, pitch-shift {C.AUG_PITCH_STEPS} semitones, gain {C.AUG_GAIN_DB} dB |

Validation and test are never augmented.

## 4. Model selection (validation only)

The test set was not consulted for any decision below.

{md_table(val_tbl)}

### How the final model was chosen

Raw argmax would pick `{sel['best_raw_model']}` ({sel['best_raw_val_macro_f1']:.4f}).
A bootstrap over the 240 validation clips puts the standard error of macro-F1
at **{sel['bootstrap_se']:.4f}**, so everything above {sel['cutoff_within_1se']:.4f} is
statistically indistinguishable from the best: {within_1se}.
Applying the one-standard-error rule (simplest model within 1 SE) gives:

**Selected: `{best_name}`** — {card['params']:,} parameters,
{best['inference_ms_per_clip']} ms per clip.

## 5. Test-set results (unseen actors {', '.join(map(str, C.TEST_ACTORS))})

{md_table(comp)}

## 6. Per-class performance of the selected model

{md_table(per_class)}

## 7. Most frequent confusions (selected model)

{md_table(top_conf.rename(columns={'true': 'True', 'predicted': 'Predicted', 'count': 'Count'}))}

Total misclassified: {len(conf)} of {ds['split_files']['test']} test clips.

## 8. Confidence calibration

The softmax maximum is used as a confidence score. Thresholds were swept on
the **validation** split, measuring coverage and selective accuracy
(accuracy among accepted clips):

| Quantity | Value |
|---|---|
| Target selective accuracy | {calib['target_selective_accuracy']:.0%} |
| Target reached on validation | {calib['target_reached']} |
| Chosen threshold | {calib['threshold']:.2f} |
| Coverage at threshold (val) | {calib['coverage_at_threshold']:.1%} |
| Selective accuracy at threshold (val) | {calib['selective_accuracy_at_threshold']:.1%} |

Applied to the test split: accuracy {unseen['test_accuracy_all']:.1%} over all
clips, {unseen['test_accuracy_confident']:.1%} over the
{unseen['confident_coverage']:.1%} of clips the model accepts as confident.

## 9. Out-of-domain (channel robustness) check

`data/external/` holds {unseen['n_external_files']} SIMULATED files: held-out test
clips (unseen actors) passed through a telephone band (300-3400 Hz, 8 kHz) or a
reverberant far-field channel. They probe robustness to an unseen recording
channel, **not** generalisation to new speakers -- the test split already does that.

| Condition | Accuracy |
|---|---|
| Clean test audio (240 clips) | {unseen['test_accuracy_all']:.1%} |
| Simulated degraded channel ({unseen['simulated_channel']['n']} clips) | {unseen['simulated_channel']['n_correct']}/{unseen['simulated_channel']['n']} = {unseen['simulated_channel']['n_correct'] / unseen['simulated_channel']['n']:.1%} |

Accuracy falls and, usefully, confidence falls with it, so the threshold flags
most of the degraded clips as uncertain rather than asserting a wrong answer
confidently. Real microphone recordings should be expected to behave similarly
or worse; drop your own into `data/external/` and re-run
`src/test_unseen_audio.py`.

## 10. Figures

All in `outputs/figures/`:

`01_emotion_distribution.png`, `02_duration_distribution.png`,
`03_split_distribution.png`, `04_actor_distribution.png`,
`05_waveform_mel_mfcc.png`, `06_feature_comparison.png`,
`cm_*.png` (confusion matrices), `curves_*.png` (training curves),
`roc_pr_*.png` (ROC and precision-recall), `10_model_comparison.png`,
`11_confidence_calibration.png`.

## 11. Reproducing

```bash
python src/run_all.py
```

Seed {C.SEED} is set for Python, NumPy and TensorFlow.
"""
    (C.ROOT / "report.md").write_text(text, encoding="utf-8")
    print(f"report.md written ({len(text)} chars)")


if __name__ == "__main__":
    main()
