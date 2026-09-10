"""Create SIMULATED out-of-domain audio for the external-audio test.

Honest framing, because this matters for how the results are read:

  * These files are NOT new speakers and NOT new content.  They are held-out
    test clips (unseen actors) passed through a degraded recording channel.
  * They therefore measure ONE thing: robustness to a channel the model never
    saw during training (band-limiting, additive noise, reverberation,
    level loss).  They do not measure generalisation to new speakers -- the
    test split already does that.
  * A genuine external recording (your own microphone) is still the better
    test.  Drop .wav files into data/external/ and re-run
    src/test_unseen_audio.py.

Anyone reading outputs/predictions/ should be able to tell these apart from
real recordings, so the filenames are prefixed with ``simulated_``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sps
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from audio_preprocessing import load_audio  # noqa: E402

OUT = C.DATA_RAW.parent / "external"


def telephone(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Narrow-band telephone channel: 300-3400 Hz, 8 kHz round trip, noise."""
    sos = sps.butter(6, [300, 3400], btype="band", fs=C.SAMPLE_RATE, output="sos")
    y = sps.sosfilt(sos, y)
    y = sps.resample_poly(y, 8000, C.SAMPLE_RATE)
    y = sps.resample_poly(y, C.SAMPLE_RATE, 8000)
    power = np.mean(y ** 2)
    y = y + rng.normal(0, np.sqrt(power / 10 ** (18 / 10)), len(y))
    return y.astype(np.float32)


def far_field(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverberant far-field capture: exponential-decay room + noise + level loss."""
    rt = int(0.25 * C.SAMPLE_RATE)
    ir = rng.normal(0, 1, rt) * np.exp(-np.linspace(0, 6, rt))
    ir[0] = 1.0
    y = sps.fftconvolve(y, ir)[:len(y)]
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.3
    power = np.mean(y ** 2)
    y = y + rng.normal(0, np.sqrt(power / 10 ** (12 / 10)), len(y))
    return y.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(C.DATA_PROCESSED / "manifest.csv")
    test = manifest[manifest.split == "test"]
    rng = np.random.default_rng(C.SEED)

    # One clip from each of four contrasting emotions, all from unseen actors.
    picks = []
    for emo in ("angry", "happy", "sad", "neutral"):
        sub = test[test.emotion == emo]
        if len(sub):
            picks.append(sub.iloc[0])

    made = []
    for r in picks:
        y = load_audio(r["path"])
        for tag, fn in (("telephone", telephone), ("farfield", far_field)):
            deg = fn(y, rng)
            name = f"simulated_{tag}_{r['emotion']}_actor{r['actor']}.wav"
            sf.write(OUT / name, deg, C.SAMPLE_RATE)
            made.append({"file": name, "true_emotion": r["emotion"],
                         "actor": int(r["actor"]), "degradation": tag,
                         "source_file": r["filename"]})
            print("wrote", name)

    pd.DataFrame(made).to_csv(OUT / "simulated_ground_truth.csv", index=False)
    (OUT / "README.txt").write_text(
        "These files are SIMULATED channel degradations of held-out RAVDESS "
        "test clips (unseen actors).\nThey test robustness to an unseen "
        "recording channel, NOT generalisation to unseen speakers or "
        "content.\nGround truth is in simulated_ground_truth.csv.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
