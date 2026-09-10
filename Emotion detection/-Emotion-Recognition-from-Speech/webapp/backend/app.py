"""Flask API for the Speech Emotion Recognition web UI.

Serves three things:
  * the exported model's predictions for uploaded audio
  * the metrics the pipeline produced (dataset summary, comparison, confusion)
  * a handful of held-out test clips so the demo works without an upload

Run:
    .venv\\Scripts\\python.exe webapp\\backend\\app.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from predict import EmotionRecognizer  # noqa: E402

DIST = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(DIST) if DIST.exists() else None)
CORS(app)

ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_BYTES = 20 * 1024 * 1024

_recognizer: EmotionRecognizer | None = None


def recognizer() -> EmotionRecognizer:
    """Load the exported bundle once, on first use."""
    global _recognizer
    if _recognizer is None:
        _recognizer = EmotionRecognizer()
    return _recognizer


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- metrics ---
@app.get("/api/overview")
def overview():
    """Everything the dashboard needs, in one round trip."""
    summary = load_json(C.METRICS_DIR / "dataset_summary.json")
    card = load_json(C.MODELS_DIR / "final_model" / "model_card.json")
    metrics = load_json(C.METRICS_DIR / "test_metrics_all.json")
    ranking = load_json(C.METRICS_DIR / "validation_ranking.json")
    calib = load_json(C.METRICS_DIR / "confidence_calibration.json")
    unseen = load_json(C.METRICS_DIR / "unseen_audio_summary.json")
    comparison = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")

    best_name = card["selected_model"]
    best = metrics[best_name]

    preds = pd.read_csv(C.PRED_DIR / "test_split_predictions.csv")
    classes = card["classes"]
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(preds["true"], preds["pred"]):
        matrix[index[t], index[p]] += 1

    mis = pd.read_csv(C.PRED_DIR / f"misclassified_{best_name}.csv")
    confusions = (mis.groupby(["true", "predicted"]).size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False).head(8)
                  .to_dict("records"))

    per_actor = (preds.groupby("actor")["correct"].agg(["mean", "count"])
                 .reset_index().to_dict("records"))

    return jsonify({
        "dataset": {
            "recordings": summary["total_recordings"],
            "speakers": summary["total_speakers"],
            "emotions": summary["total_emotions"],
            "classDistribution": summary["class_distribution"],
            "duration": summary["duration_sec"],
            "splitActors": summary["split_actors"],
            "splitFiles": summary["split_files"],
            "samplingRates": summary["sampling_rates"],
            "corrupt": summary["corrupt_or_unreadable"],
        },
        "model": {
            "name": best_name,
            "architecture": card["architecture"],
            "params": card["params"],
            "classes": classes,
            "threshold": card["confidence_threshold"],
            "selectionRule": card["selection_details"]["rule"],
            "selection": card["selection_details"],
            "epochs": card["epochs_run"],
            "trainSeconds": card["train_seconds"],
        },
        "test": {
            "accuracy": best["accuracy"],
            "macroF1": best["f1_macro"],
            "weightedF1": best["f1_weighted"],
            "precisionMacro": best["precision_macro"],
            "recallMacro": best["recall_macro"],
            "rocAuc": best["roc_auc_ovr_macro"],
            "msPerClip": best["inference_ms_per_clip"],
            "perClass": best["per_class"],
        },
        "comparison": comparison.replace({np.nan: None}).to_dict("records"),
        "validationRanking": ranking,
        "confusionMatrix": matrix.tolist(),
        "topConfusions": confusions,
        "perActor": per_actor,
        "calibration": {k: v for k, v in calib.items() if k != "sweep"},
        "calibrationSweep": calib["sweep"],
        "unseen": unseen,
    })


# ---------------------------------------------------------------- samples ---
def sample_index() -> list[dict]:
    """One held-out test clip per emotion, plus the degraded-channel files."""
    manifest = pd.read_csv(C.DATA_PROCESSED / "manifest.csv")
    test = manifest[manifest.split == "test"]
    items = []
    for emo in sorted(test["emotion"].unique()):
        r = test[test.emotion == emo].iloc[0]
        items.append({
            "id": r["filename"],
            "label": r["emotion"],
            "actor": int(r["actor"]),
            "gender": r["gender"],
            "intensity": r["intensity"],
            "duration": round(float(r["duration"]), 2),
            "kind": "test",
            "path": r["path"],
        })
    ext_dir = C.DATA_RAW.parent / "external"
    gt_file = ext_dir / "simulated_ground_truth.csv"
    if gt_file.exists():
        for row in pd.read_csv(gt_file).to_dict("records"):
            p = ext_dir / row["file"]
            if p.exists():
                items.append({
                    "id": row["file"], "label": row["true_emotion"],
                    "actor": int(row["actor"]), "gender": "",
                    "intensity": row["degradation"], "duration": None,
                    "kind": "degraded", "path": str(p),
                })
    return items


@app.get("/api/samples")
def samples():
    return jsonify([{k: v for k, v in s.items() if k != "path"}
                    for s in sample_index()])


@app.get("/api/audio/<path:sample_id>")
def audio(sample_id: str):
    match = next((s for s in sample_index() if s["id"] == sample_id), None)
    if match is None:
        return jsonify({"error": "unknown sample"}), 404
    return send_file(match["path"], mimetype="audio/wav")


@app.post("/api/predict/sample/<path:sample_id>")
def predict_sample(sample_id: str):
    match = next((s for s in sample_index() if s["id"] == sample_id), None)
    if match is None:
        return jsonify({"error": "unknown sample"}), 404
    result = recognizer().predict(match["path"])
    result["trueLabel"] = match["label"]
    result["source"] = match["kind"]
    return jsonify(result)


# ---------------------------------------------------------------- predict ---
@app.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send multipart field 'file'."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": f"Unsupported type '{suffix}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED))}"}), 400

    blob = f.read()
    if len(blob) > MAX_BYTES:
        return jsonify({"error": "File too large (limit 20 MB)."}), 400
    if not blob:
        return jsonify({"error": "File is empty."}), 400

    tmp = Path(tempfile.gettempdir()) / f"ser_upload{suffix}"
    tmp.write_bytes(blob)
    try:
        result = recognizer().predict(tmp)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not decode audio: {exc}"}), 400
    finally:
        tmp.unlink(missing_ok=True)

    result["audio"] = f.filename
    result["source"] = "upload"
    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok",
                    "modelLoaded": _recognizer is not None,
                    "bundle": str(C.MODELS_DIR / "final_model")})


# ------------------------------------------------------ static frontend -----
@app.get("/")
@app.get("/<path:page>")
def spa(page: str = ""):
    if not DIST.exists():
        return jsonify({"error": "Frontend not built. Run `npm run build` "
                                 "in webapp/frontend."}), 404
    target = DIST / page
    if page and target.exists() and target.is_file():
        return send_from_directory(str(DIST), page)
    return send_from_directory(str(DIST), "index.html")


if __name__ == "__main__":
    print("Loading model bundle...")
    recognizer()
    print("Ready on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
