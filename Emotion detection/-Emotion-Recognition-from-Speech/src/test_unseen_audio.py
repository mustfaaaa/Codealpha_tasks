"""PHASE 24 -- reload the exported model and predict on genuinely unseen audio.

Deliberately keeps two things apart:

  DATASET TEST PERFORMANCE   -- clips from the four held-out RAVDESS actors,
                                same recording conditions as training.
  EXTERNAL AUDIO PERFORMANCE -- any file the user drops into
                                data/external/; different microphone, room
                                and speaker, so a drop in accuracy here is
                                expected and is the honest real-world number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from predict import EmotionRecognizer, format_result  # noqa: E402

EXTERNAL_DIR = C.DATA_RAW.parent / "external"


def main() -> None:
    rec = EmotionRecognizer()
    print(f"Reloaded bundle: {rec.card['selected_model']} "
          f"({rec.card['architecture']}), {len(rec.classes)} classes, "
          f"confidence threshold {rec.threshold:.2f}\n")

    manifest = pd.read_csv(C.DATA_PROCESSED / "manifest.csv")
    rows = []

    # One clip per split, so the difference between seen and unseen speakers
    # is visible side by side.
    print("=" * 66)
    print("A. SANITY CHECK — one clip from each split")
    print("=" * 66)
    for split in ("train", "val", "test"):
        r = manifest[manifest.split == split].sample(1, random_state=C.SEED).iloc[0]
        res = rec.predict(r["path"])
        print(f"[{split}] actor {r['actor']} — true label: {r['emotion']}")
        print(format_result(res))
        print("-" * 66)
        rows.append({"source": f"dataset:{split}", "file": r["filename"],
                     "true": r["emotion"], "pred": res["predicted_emotion"],
                     "confidence": res["confidence"],
                     "correct": res["predicted_emotion"] == r["emotion"]})

    # Whole held-out test split: the headline speaker-independent number.
    print("\n" + "=" * 66)
    print("B. DATASET TEST PERFORMANCE — all 240 clips, 4 unseen actors")
    print("=" * 66)
    te = manifest[manifest.split == "test"]
    preds = [rec.predict(p) for p in te["path"]]
    tdf = pd.DataFrame({
        "filename": te["filename"].values,
        "actor": te["actor"].values,
        "true": te["emotion"].values,
        "pred": [p["predicted_emotion"] for p in preds],
        "confidence": [p["confidence"] for p in preds],
        "reliable": [p["reliable"] for p in preds],
    })
    tdf["correct"] = tdf["true"] == tdf["pred"]
    print(f"accuracy (all clips)          : {tdf['correct'].mean():.4f}")
    acc = tdf[tdf["reliable"]]
    print(f"accuracy (confident clips)    : "
          f"{acc['correct'].mean():.4f}  on {len(acc)}/{len(tdf)} clips "
          f"({len(acc) / len(tdf):.1%} coverage)")
    print("\nper-actor accuracy:")
    print(tdf.groupby("actor")["correct"].agg(["mean", "count"]).to_string())
    print("\nper-emotion accuracy:")
    print(tdf.groupby("true")["correct"].agg(["mean", "count"]).to_string())
    tdf.to_csv(C.PRED_DIR / "test_split_predictions.csv", index=False)

    # External audio, if the user supplied any.
    print("\n" + "=" * 66)
    print("C. EXTERNAL AUDIO PERFORMANCE — files in data/external/")
    print("=" * 66)
    ext = sorted([p for p in EXTERNAL_DIR.glob("*")
                  if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg",
                                          ".m4a"}]) if EXTERNAL_DIR.exists() else []
    if not ext:
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"No external recordings found in {EXTERNAL_DIR}.")
        print("Drop your own .wav/.mp3 files there and re-run this script to")
        print("measure real-world performance. Dataset numbers above do NOT")
        print("transfer directly to phone-recorded audio.")
    else:
        gt_path = EXTERNAL_DIR / "simulated_ground_truth.csv"
        gt = (pd.read_csv(gt_path).set_index("file")["true_emotion"].to_dict()
              if gt_path.exists() else {})
        for p in ext:
            res = rec.predict(p)
            print(format_result(res))
            true = gt.get(p.name)
            if true:
                print(f"true label (simulated clip): {true}  -> "
                      f"{'CORRECT' if true == res['predicted_emotion'] else 'WRONG'}")
            print("-" * 66)
            rows.append({"source": "external", "file": p.name, "true": true,
                         "pred": res["predicted_emotion"],
                         "confidence": res["confidence"],
                         "reliable": res["reliable"],
                         "correct": (None if not true
                                     else true == res["predicted_emotion"])})
        scored = [r for r in rows if r["source"] == "external"
                  and r["correct"] is not None]
        if scored:
            n_ok = sum(r["correct"] for r in scored)
            mean_conf = sum(r["confidence"] for r in scored) / len(scored)
            n_conf = sum(r["reliable"] for r in scored)
            print(f"\nSimulated-channel accuracy: {n_ok}/{len(scored)} "
                  f"= {n_ok / len(scored):.1%}   mean confidence "
                  f"{mean_conf:.1%}   flagged confident: {n_conf}/{len(scored)}")
            print("These are held-out test clips passed through a degraded")
            print("recording channel: they measure channel robustness, not")
            print("generalisation to new speakers.")

    pd.DataFrame(rows).to_csv(C.PRED_DIR / "unseen_audio_demo.csv", index=False)
    summary = {
        "model": rec.card["selected_model"],
        "test_accuracy_all": float(tdf["correct"].mean()),
        "test_accuracy_confident": float(acc["correct"].mean()) if len(acc) else None,
        "confident_coverage": float(len(acc) / len(tdf)),
        "n_external_files": len(ext),
        "simulated_channel": ({
            "n": len([r for r in rows if r["source"] == "external" and r["correct"] is not None]),
            "n_correct": int(sum(r["correct"] for r in rows
                                 if r["source"] == "external" and r["correct"] is not None)),
        } if any(r["source"] == "external" and r["correct"] is not None for r in rows) else None),
    }
    (C.METRICS_DIR / "unseen_audio_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
