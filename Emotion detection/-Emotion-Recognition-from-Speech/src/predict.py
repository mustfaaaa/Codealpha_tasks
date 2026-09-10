"""PHASES 22-23 -- inference pipeline with calibrated confidence handling.

Loads the exported bundle from models/final_model/ and predicts the emotion of
any audio file.  The bundle carries the preprocessing and feature-extraction
configuration that was used at training time; this module asserts that the
current code still matches it, so a silent configuration drift becomes a loud
error instead of quietly wrong predictions.

CLI:
    python src/predict.py path/to/audio.wav
    python src/predict.py a.wav b.wav --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from feature_extraction import features_from_file  # noqa: E402

BUNDLE = C.MODELS_DIR / "final_model"


class EmotionRecognizer:
    """Reloadable speech-emotion predictor."""

    def __init__(self, bundle_dir: Path = BUNDLE):
        import tensorflow as tf  # imported lazily: keeps --help fast

        self.dir = Path(bundle_dir)
        self.card = json.loads((self.dir / "model_card.json").read_text())
        self.cfg = json.loads((self.dir / "feature_config.json").read_text())
        self.norm = json.loads((self.dir / "norm_stats.json").read_text())
        self.le = joblib.load(self.dir / "label_encoder.joblib")
        self.classes = list(self.le.classes_)
        self.view = self.card["feature_view"]
        self.threshold = self.card["confidence_threshold"]
        self._check_config()
        self.model = tf.keras.models.load_model(self.dir / "model.keras")

    def _check_config(self) -> None:
        """Guarantee inference-time settings equal training-time settings."""
        live = {
            "sample_rate": C.SAMPLE_RATE, "duration_sec": C.DURATION,
            "trim_top_db": C.TRIM_TOP_DB, "peak_normalize": C.PEAK_NORM,
            "n_mfcc": C.N_MFCC, "n_fft": C.N_FFT, "hop_length": C.HOP_LENGTH,
            "win_length": C.WIN_LENGTH, "n_mels": C.N_MELS,
            "fmin": C.FMIN, "fmax": C.FMAX, "n_frames": C.N_FRAMES,
        }
        mismatched = {k: (v, self.cfg.get(k)) for k, v in live.items()
                      if self.cfg.get(k) != v}
        if mismatched:
            raise RuntimeError(
                "Feature configuration drift between training bundle and "
                f"current code: {mismatched}")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.view == "cnn":
            m = np.asarray(self.norm["cnn_mean"], dtype=np.float32)
            s = np.asarray(self.norm["cnn_std"], dtype=np.float32)
            return ((x - m) / s).astype(np.float32)
        m = np.asarray(self.norm["seq_mean"], dtype=np.float32)
        return ((x - m) / self.norm["seq_std"]).astype(np.float32)

    def predict(self, audio_path: str | Path) -> dict:
        """Return the predicted emotion, confidence and every class probability."""
        feats = features_from_file(audio_path)
        x = self._normalize(feats[self.view])[None, ...]
        prob = self.model.predict(x, verbose=0)[0]
        idx = int(prob.argmax())
        conf = float(prob[idx])
        order = np.argsort(prob)[::-1]
        return {
            "audio": str(audio_path),
            "predicted_emotion": self.classes[idx],
            "confidence": conf,
            "reliable": bool(conf >= self.threshold),
            "status": ("confident" if conf >= self.threshold
                       else "LOW CONFIDENCE - emotion uncertain"),
            "confidence_threshold": self.threshold,
            "probabilities": {self.classes[i]: float(prob[i]) for i in order},
        }

    def predict_batch(self, paths) -> list[dict]:
        return [self.predict(p) for p in paths]


def format_result(r: dict) -> str:
    lines = [
        f"Audio              : {Path(r['audio']).name}",
        f"Predicted Emotion  : {r['predicted_emotion'].upper()}",
        f"Confidence         : {r['confidence'] * 100:.1f}%",
        f"Status             : {r['status']}"
        + ("" if r["reliable"] else
           f"  (threshold {r['confidence_threshold'] * 100:.1f}%)"),
        "",
        "Emotion probabilities:",
    ]
    for emo, p in r["probabilities"].items():
        bar = "#" * int(round(p * 40))
        lines.append(f"  {emo:<10} {p * 100:6.2f}%  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict emotion from speech audio")
    ap.add_argument("audio", nargs="+", help="one or more audio files")
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--json", action="store_true", help="emit JSON instead")
    a = ap.parse_args()

    rec = EmotionRecognizer(Path(a.bundle))
    results = rec.predict_batch(a.audio)
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"[model: {rec.card['selected_model']}  "
              f"classes: {len(rec.classes)}]\n")
        for r in results:
            print(format_result(r))
            print("-" * 62)


if __name__ == "__main__":
    main()
