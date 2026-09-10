"""Feature extraction.

Three representations are produced from the *same* preprocessed waveform so
that all models see identical audio:

1. ``mfcc_stack``  (40, 94, 3)  -> 2-D CNN input   [MFCC, delta, delta-delta]
2. ``sequence``    (94, 120)    -> LSTM input      [per-frame concatenation]
3. ``stats``       (fixed vec)  -> classical ML baseline

Why MFCCs for emotion?  The mel filterbank mimics human frequency resolution,
and the DCT that follows decorrelates the log-mel energies into a compact set
of coefficients describing the *spectral envelope* -- i.e. vocal-tract shape
and voice quality.  Emotion changes exactly those things (tense vs. breathy
phonation, formant shifts, spectral tilt), which is why MFCCs plus their
temporal derivatives remain the standard front-end for speech emotion
recognition.  The deltas add the rate of change, capturing prosodic dynamics
that a static frame cannot express.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from audio_preprocessing import preprocess  # noqa: E402


def mfcc_stack(y: np.ndarray) -> np.ndarray:
    """(n_mfcc, n_frames, 3) tensor of MFCC + delta + delta-delta."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=C.SAMPLE_RATE, n_mfcc=C.N_MFCC, n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH, win_length=C.WIN_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX,
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    stack = np.stack([mfcc, d1, d2], axis=-1)
    assert stack.shape == C.CNN_INPUT_SHAPE, f"got {stack.shape}"
    return stack.astype(np.float32)


def to_sequence(stack: np.ndarray) -> np.ndarray:
    """(n_frames, n_mfcc*3) view of the same tensor for recurrent models."""
    n_mfcc, n_frames, n_ch = stack.shape
    return stack.transpose(1, 0, 2).reshape(n_frames, n_mfcc * n_ch)


def stat_features(y: np.ndarray) -> np.ndarray:
    """Fixed-length descriptor for the classical baseline.

    Time-aggregates (mean + std) of MFCC/delta/delta2 plus a small set of
    complementary low-level descriptors that capture information MFCCs
    compress away: noisiness (ZCR), brightness (spectral centroid), spread
    (bandwidth), spectral shape (rolloff, flatness), harmony (chroma) and
    loudness dynamics (RMS).
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=C.SAMPLE_RATE, n_mfcc=C.N_MFCC, n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH, win_length=C.WIN_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX,
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    S = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                            win_length=C.WIN_LENGTH))
    extras = [
        librosa.feature.zero_crossing_rate(y, hop_length=C.HOP_LENGTH),
        librosa.feature.spectral_centroid(S=S, sr=C.SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(S=S, sr=C.SAMPLE_RATE),
        librosa.feature.spectral_rolloff(S=S, sr=C.SAMPLE_RATE),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.rms(S=S),
        librosa.feature.chroma_stft(S=S, sr=C.SAMPLE_RATE),
    ]
    blocks = [mfcc, d1, d2] + extras
    vec = np.concatenate(
        [np.concatenate([b.mean(axis=1), b.std(axis=1)]) for b in blocks]
    )
    return vec.astype(np.float32)


def features_from_waveform(y: np.ndarray) -> dict[str, np.ndarray]:
    stack = mfcc_stack(y)
    return {"cnn": stack, "seq": to_sequence(stack), "stats": stat_features(y)}


def features_from_file(path: str | Path) -> dict[str, np.ndarray]:
    """Preprocess a file then extract all three representations."""
    return features_from_waveform(preprocess(path))


FEATURE_NAMES_NOTE = (
    "stats vector = [mfcc(40) d1(40) d2(40)] x (mean,std) + "
    "[zcr, centroid, bandwidth, rolloff, flatness, rms](1 each) x (mean,std) + "
    "chroma(12) x (mean,std)"
)
