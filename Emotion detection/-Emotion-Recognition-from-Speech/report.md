# Speech Emotion Recognition on RAVDESS — Project Report

Generated from executed pipeline outputs. Every figure below is read from
`outputs/metrics/`; none is hand-written.

## 1. Dataset

**RAVDESS**, speech subset (Zenodo record 1188976,
`Audio_Speech_Actors_01-24.zip`, CC BY-NC-SA 4.0).

| Property | Value |
|---|---|
| Total recordings | 1440 |
| Speakers (actors) | 24 (720 male / 720 female clips) |
| Emotions | 8 |
| Native sampling rate | 48000 Hz x 1440 |
| Channels | 1ch x 1435, 2ch x 5 |
| Duration min / mean / max | 2.94s / 3.70s / 5.27s |
| Duration median (std) | 3.67s (0.34s) |
| Corrupt / unreadable | 0 |
| Missing files | 0 |
| Byte-identical duplicates | 2 (03-01-03-01-02-01-07.wav, 03-01-03-01-02-02-07.wav) |
| Class imbalance (max/min) | 2.0x |

### Class distribution

| Emotion | Recordings |
|---|---|
| neutral | 96 |
| calm | 192 |
| happy | 192 |
| sad | 192 |
| angry | 192 |
| fearful | 192 |
| disgust | 192 |
| surprised | 192 |

Only `neutral` is under-represented: RAVDESS records neutral at normal
intensity only, while every other emotion is recorded at both normal and
strong intensity. This 2:1 imbalance is handled with class weights, and macro-F1
is used as the primary metric so the small class cannot be ignored for free.

## 2. Speaker-independent split (leakage prevention)

Actors are **disjoint** across partitions, and every partition is gender
balanced (odd actor id = male, even = female):

| Split | Actors | Clips |
|---|---|---|
| train | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24 | 960 |
| val | 16, 17, 18, 19 | 240 |
| test | 20, 21, 22, 23 | 240 |

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
| Sampling rate | 16000 Hz |
| Silence trimming | leading/trailing only, top_db=30 |
| Amplitude | peak normalisation |
| Length handling | centre crop / symmetric zero-pad to 3.0s (48000 samples) |
| MFCC | n_mfcc=40, n_fft=2048, hop=512, win=2048, n_mels=128, fmin=20, fmax=8000 |
| Derived channels | MFCC, delta, delta-delta |
| CNN tensor | (40, 94, 3) |
| Sequence tensor | (94, 120) |
| Standardisation | per-channel, statistics from the **training split only** |
| Augmentation (train only) | 2 extra copies: noise (SNR 15-30 dB), time-stretch (0.9, 1.1), pitch-shift (-2.0, 2.0) semitones, gain (-6.0, 6.0) dB |

Validation and test are never augmented.

## 4. Model selection (validation only)

The test set was not consulted for any decision below.

| Model | Val Accuracy | Val Macro F1 | Params | Epochs | Train (s) |
|---|---|---|---|---|---|
| cnn_t2 | 0.6833 | 0.6609 | 1,216,840 | 48 | 2939.7 |
| cnn_aug | 0.6708 | 0.6502 | 306,344 | 36 | 963.2 |
| cnn_t1 | 0.6708 | 0.6502 | 306,344 | 36 | 701.5 |
| cnn_t3 | 0.6375 | 0.6233 | 75,688 | 40 | 754.4 |
| cnnlstm_aug | 0.5958 | 0.5919 | 1,054,728 | 30 | 1262.0 |
| lstm_aug | 0.5292 | 0.5157 | 428,104 | 32 | 583.3 |
| cnn_noaug | 0.2917 | 0.2254 | 306,344 | 34 | 307.3 |

### How the final model was chosen

Raw argmax would pick `cnn_t2` (0.6609).
A bootstrap over the 240 validation clips puts the standard error of macro-F1
at **0.0307**, so everything above 0.6302 is
statistically indistinguishable from the best: `cnn_aug`, `cnn_t1`, `cnn_t2`.
Applying the one-standard-error rule (simplest model within 1 SE) gives:

**Selected: `cnn_aug`** — 306,344 parameters,
2.534 ms per clip.

## 5. Test-set results (unseen actors 20, 21, 22, 23)

| Model | Accuracy | Precision(macro) | Recall(macro) | Macro F1 | Weighted F1 | ROC-AUC(ovr) | Params | ms/clip |
|---|---|---|---|---|---|---|---|---|
| cnn_t2 | 0.6958 | 0.7199 | 0.7109 | 0.7007 | 0.6974 | 0.9521 | 1216840.0 | 9.367 |
| cnn_aug | 0.7 | 0.6986 | 0.7031 | 0.6905 | 0.6894 | 0.9416 | 306344.0 | 2.534 |
| cnn_t1 | 0.7 | 0.6986 | 0.7031 | 0.6905 | 0.6894 | 0.9416 | 306344.0 | 5.519 |
| cnn_t3 | 0.6333 | 0.6214 | 0.6367 | 0.6209 | 0.6246 | 0.9297 | 75688.0 | 2.288 |
| cnnlstm_aug | 0.5292 | 0.5322 | 0.5352 | 0.5158 | 0.5184 | 0.8562 | 1054728.0 | 9.023 |
| baseline_logreg | 0.4542 | 0.4623 | 0.457 | 0.4514 | 0.4518 | 0.8509 |  | 0.0 |
| lstm_aug | 0.425 | 0.4437 | 0.418 | 0.4106 | 0.4142 | 0.8148 | 428104.0 | 6.61 |
| baseline_svm_rbf | 0.4167 | 0.4192 | 0.4062 | 0.4022 | 0.4118 | 0.8307 |  | 0.401 |
| baseline_random_forest | 0.4292 | 0.4063 | 0.4219 | 0.3927 | 0.4022 | 0.8343 |  | 0.363 |
| cnn_noaug | 0.3333 | 0.547 | 0.3398 | 0.2913 | 0.2874 | 0.8463 | 306344.0 | 2.904 |

## 6. Per-class performance of the selected model

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| disgust | 0.938 | 0.938 | 0.938 | 32 |
| calm | 0.737 | 0.875 | 0.8 | 32 |
| surprised | 0.651 | 0.875 | 0.747 | 32 |
| sad | 0.686 | 0.75 | 0.716 | 32 |
| neutral | 0.667 | 0.75 | 0.706 | 16 |
| angry | 0.789 | 0.469 | 0.588 | 32 |
| fearful | 0.576 | 0.594 | 0.585 | 32 |
| happy | 0.545 | 0.375 | 0.444 | 32 |

## 7. Most frequent confusions (selected model)

| True | Predicted | Count |
|---|---|---|
| angry | surprised | 8 |
| happy | surprised | 7 |
| sad | calm | 7 |
| fearful | sad | 6 |
| happy | fearful | 6 |
| fearful | happy | 5 |
| happy | neutral | 5 |
| angry | fearful | 4 |
| calm | sad | 4 |
| neutral | calm | 3 |

Total misclassified: 72 of 240 test clips.

## 8. Confidence calibration

The softmax maximum is used as a confidence score. Thresholds were swept on
the **validation** split, measuring coverage and selective accuracy
(accuracy among accepted clips):

| Quantity | Value |
|---|---|
| Target selective accuracy | 80% |
| Target reached on validation | True |
| Chosen threshold | 0.82 |
| Coverage at threshold (val) | 57.1% |
| Selective accuracy at threshold (val) | 81.0% |

Applied to the test split: accuracy 70.0% over all
clips, 77.8% over the
63.7% of clips the model accepts as confident.

## 9. Out-of-domain (channel robustness) check

`data/external/` holds 8 SIMULATED files: held-out test
clips (unseen actors) passed through a telephone band (300-3400 Hz, 8 kHz) or a
reverberant far-field channel. They probe robustness to an unseen recording
channel, **not** generalisation to new speakers -- the test split already does that.

| Condition | Accuracy |
|---|---|
| Clean test audio (240 clips) | 70.0% |
| Simulated degraded channel (8 clips) | 5/8 = 62.5% |

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

Seed 42 is set for Python, NumPy and TensorFlow.
