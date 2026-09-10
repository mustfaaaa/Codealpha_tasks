"""End-to-end driver: runs every stage of the project in order.

    python src/run_all.py            # full pipeline
    python src/run_all.py --skip-features   # reuse cached .npz features
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], label: str) -> None:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {label} (exit {r.returncode})")
    print(f"--- {label} finished in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    a = ap.parse_args()

    if not a.skip_features:
        run(["src/analyze_dataset.py"], "PHASE 3  dataset analysis + figures")
        run(["src/build_features.py"], "PHASE 6  feature extraction")

    run(["src/train_baseline.py"], "PHASE 9  classical baselines")
    run(["src/train.py", "--model", "cnn", "--no-aug", "--tag", "cnn_noaug"],
        "PHASE 10 CNN without augmentation")
    run(["src/train.py", "--model", "cnn", "--aug", "--tag", "cnn_aug"],
        "PHASE 10 CNN with augmentation")
    run(["src/train.py", "--model", "lstm", "--aug", "--tag", "lstm_aug"],
        "PHASE 11 BiLSTM")
    run(["src/train.py", "--model", "cnn_lstm", "--aug", "--tag", "cnnlstm_aug"],
        "PHASE 12 CNN-LSTM")

    if not a.skip_tuning:
        run(["src/tune.py", "--model", "cnn"], "PHASE 18 tuning")

    run(["src/finalize.py"], "PHASE 20 selection + calibration + export")
    run(["src/evaluate.py"], "PHASE 15 test-set evaluation")
    run(["src/make_external_demo.py"], "PHASE 24 simulated degraded-channel clips")
    run(["src/test_unseen_audio.py"], "PHASE 24 reload + unseen audio")
    run(["src/make_report.py"], "PHASE 22 build report.md")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
