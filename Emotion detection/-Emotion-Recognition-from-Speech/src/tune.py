"""PHASE 18 -- small, honest hyper-parameter search on the strongest model.

Deliberately kept to a handful of configurations: with ~960 training
utterances a large search would mostly be fitting noise in the validation
split, and every extra configuration is another chance to over-select.

The TEST SET IS NOT TOUCHED HERE -- every decision uses validation macro-F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from train import train_model  # noqa: E402

GRIDS = {
    "cnn": [
        {"tag": "cnn_t1", "lr": 5e-4, "batch_size": 32,
         "build_kwargs": {"filters": (32, 64, 128), "dropout": 0.3, "dense_units": 128}},
        {"tag": "cnn_t2", "lr": 1e-3, "batch_size": 64,
         "build_kwargs": {"filters": (64, 128, 256), "dropout": 0.4, "dense_units": 256}},
        {"tag": "cnn_t3", "lr": 5e-4, "batch_size": 32,
         "build_kwargs": {"filters": (32, 64), "dropout": 0.25, "dense_units": 128}},
    ],
    "lstm": [
        {"tag": "lstm_t1", "lr": 5e-4, "batch_size": 32,
         "build_kwargs": {"units": (128, 64), "dropout": 0.3}},
        {"tag": "lstm_t2", "lr": 1e-3, "batch_size": 64,
         "build_kwargs": {"units": (256, 128), "dropout": 0.4}},
        {"tag": "lstm_t3", "lr": 5e-4, "batch_size": 32,
         "build_kwargs": {"units": (64,), "dropout": 0.25}},
    ],
    "cnn_lstm": [
        {"tag": "cnnlstm_t1", "lr": 5e-4, "batch_size": 32,
         "build_kwargs": {"filters": (32, 64, 128), "lstm_units": 128, "dropout": 0.3}},
        {"tag": "cnnlstm_t2", "lr": 1e-3, "batch_size": 64,
         "build_kwargs": {"filters": (32, 64), "lstm_units": 64, "dropout": 0.35}},
        {"tag": "cnnlstm_t3", "lr": 3e-4, "batch_size": 32,
         "build_kwargs": {"filters": (64, 128), "lstm_units": 192, "dropout": 0.4}},
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(GRIDS))
    a = ap.parse_args()

    results = []
    for cfg in GRIDS[a.model]:
        print(f"\n=== {cfg['tag']} ===")
        results.append(train_model(
            a.model, use_aug=True, tag=cfg["tag"], lr=cfg["lr"],
            batch_size=cfg["batch_size"], build_kwargs=cfg["build_kwargs"],
            quiet=True))

    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    (C.METRICS_DIR / f"tuning_{a.model}.json").write_text(
        json.dumps(results, indent=2))
    print("\nTuning ranking (validation macro-F1):")
    for r in results:
        print(f"  {r['name']:<14} macroF1={r['val_macro_f1']:.4f} "
              f"acc={r['val_accuracy']:.4f} params={r['params']:,}")


if __name__ == "__main__":
    main()
