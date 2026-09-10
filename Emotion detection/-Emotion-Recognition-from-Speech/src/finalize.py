"""PHASES 20-21, 23 -- final model selection, confidence calibration, export.

Selection uses VALIDATION results only.  The test set is read exactly once,
afterwards, to report the performance of the already-chosen model.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from models import FEATURE_VIEW  # noqa: E402
from train import load_pack  # noqa: E402

TARGET_SELECTIVE_ACCURACY = 0.80  # accuracy we require among accepted predictions


def bootstrap_macro_f1_se(y_true: np.ndarray, prob: np.ndarray,
                          n_boot: int = 1000, seed: int = C.SEED) -> float:
    """Standard error of validation macro-F1, by bootstrap resampling clips."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    pred = prob.argmax(axis=1)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(f1_score(y_true[idx], pred[idx], average="macro",
                               zero_division=0))
    return float(np.std(scores))


def select_model(results: list[dict], y_va: np.ndarray) -> tuple[dict, dict]:
    """One-standard-error rule on validation macro-F1.

    Picking the raw argmax over a 240-clip validation split rewards noise: a
    0.01 macro-F1 difference is roughly two clips.  So we take the best score,
    estimate its standard error by bootstrap, and then choose the model with
    the FEWEST parameters whose score is within one SE of the best.  That is
    the standard 'one-standard-error rule', and it prefers the simpler model
    unless a larger one is decisively better.
    """
    best_raw = max(results, key=lambda r: r["val_macro_f1"])
    prob = np.load(C.MODELS_DIR / best_raw["model_key"] /
                   f"{best_raw['name']}_val_prob.npy")
    se = bootstrap_macro_f1_se(y_va, prob)
    cutoff = best_raw["val_macro_f1"] - se
    within = [r for r in results if r["val_macro_f1"] >= cutoff]
    chosen = min(within, key=lambda r: r["params"])
    info = {
        "best_raw_model": best_raw["name"],
        "best_raw_val_macro_f1": best_raw["val_macro_f1"],
        "bootstrap_se": se,
        "cutoff_within_1se": cutoff,
        "models_within_1se": [r["name"] for r in within],
        "selected": chosen["name"],
        "rule": "one-standard-error rule on validation macro-F1; "
                "fewest parameters among models within 1 SE of the best",
    }
    return chosen, info


def collect_val_results() -> list[dict]:
    """Every trained deep model with its validation metrics."""
    out = []
    for key in ("cnn", "lstm", "cnn_lstm"):
        d = C.MODELS_DIR / key
        if d.exists():
            for f in sorted(d.glob("*_result.json")):
                out.append(json.loads(f.read_text()))
    return out


def calibrate_threshold(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Pick a confidence threshold from VALIDATION predictions.

    The softmax maximum is used as a confidence score.  We sweep candidate
    thresholds and, for each, measure two quantities on validation data:

      coverage           -- fraction of clips whose confidence clears it
      selective accuracy -- accuracy restricted to those accepted clips

    The reported threshold is the smallest one that reaches the target
    selective accuracy, i.e. the most permissive setting that still meets the
    reliability requirement.  Nothing here is invented: both curves come from
    real validation predictions, and if no threshold reaches the target we say
    so rather than pretending.
    """
    conf = prob.max(axis=1)
    correct = prob.argmax(axis=1) == y_true
    grid = np.round(np.arange(0.20, 0.96, 0.01), 2)
    sweep = []
    for t in grid:
        acc_mask = conf >= t
        cov = float(acc_mask.mean())
        sel = float(correct[acc_mask].mean()) if acc_mask.sum() else float("nan")
        sweep.append({"threshold": float(t), "coverage": cov,
                      "selective_accuracy": sel,
                      "n_accepted": int(acc_mask.sum())})

    feasible = [s for s in sweep
                if s["n_accepted"] >= 20
                and s["selective_accuracy"] >= TARGET_SELECTIVE_ACCURACY]
    if feasible:
        chosen = min(feasible, key=lambda s: s["threshold"])
        reached = True
    else:
        # Target unreachable on this dataset -- fall back to the threshold with
        # the best selective accuracy that still keeps at least half the data.
        usable = [s for s in sweep if s["coverage"] >= 0.5]
        chosen = max(usable, key=lambda s: s["selective_accuracy"])
        reached = False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([s["threshold"] for s in sweep],
            [s["selective_accuracy"] for s in sweep],
            label="selective accuracy (accepted clips)")
    ax.plot([s["threshold"] for s in sweep],
            [s["coverage"] for s in sweep], label="coverage")
    ax.axhline(TARGET_SELECTIVE_ACCURACY, ls=":", color="grey",
               label=f"target {TARGET_SELECTIVE_ACCURACY:.0%}")
    ax.axvline(chosen["threshold"], ls="--", color="crimson",
               label=f"chosen threshold = {chosen['threshold']:.2f}")
    ax.set(xlabel="softmax confidence threshold", ylabel="value",
           title="Confidence calibration on the VALIDATION split")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "11_confidence_calibration.png", dpi=140)
    plt.close(fig)

    return {"threshold": chosen["threshold"],
            "coverage_at_threshold": chosen["coverage"],
            "selective_accuracy_at_threshold": chosen["selective_accuracy"],
            "target_selective_accuracy": TARGET_SELECTIVE_ACCURACY,
            "target_reached": reached,
            "sweep": sweep}


def main() -> None:
    results = collect_val_results()
    if not results:
        raise SystemExit("No trained models found -- run src/train.py first.")

    # ---- selection on VALIDATION macro-F1 -------------------------------
    ranked = sorted(results, key=lambda r: (r["val_macro_f1"], -r["params"]),
                    reverse=True)
    print("Validation ranking (macro-F1):")
    for r in ranked:
        print(f"  {r['name']:<18} macroF1={r['val_macro_f1']:.4f} "
              f"acc={r['val_accuracy']:.4f} params={r['params']:,} "
              f"time={r['train_seconds']}s")

    va = load_pack("val")
    tr = load_pack("train_clean")
    le = LabelEncoder().fit(sorted(set(tr["label"].tolist())))
    y_va = le.transform(va["label"])

    best, sel_info = select_model(results, y_va)
    print(f"\nBest raw score : {sel_info['best_raw_model']} "
          f"({sel_info['best_raw_val_macro_f1']:.4f})")
    print(f"Bootstrap SE   : {sel_info['bootstrap_se']:.4f} "
          f"-> within-1SE cutoff {sel_info['cutoff_within_1se']:.4f}")
    print(f"Within 1 SE    : {', '.join(sel_info['models_within_1se'])}")
    print(f"Selected       : {best['name']} "
          f"({best['params']:,} params) — simplest model within 1 SE")

    view = FEATURE_VIEW[best["model_key"]]
    val_prob = np.load(C.MODELS_DIR / best["model_key"] /
                       f"{best['name']}_val_prob.npy")

    calib = calibrate_threshold(y_va, val_prob)
    print(f"Calibrated confidence threshold: {calib['threshold']:.2f} "
          f"(coverage {calib['coverage_at_threshold']:.1%}, selective accuracy "
          f"{calib['selective_accuracy_at_threshold']:.1%}, "
          f"target reached: {calib['target_reached']})")

    # ---- export bundle ---------------------------------------------------
    out = C.MODELS_DIR / "final_model"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(C.MODELS_DIR / best["model_key"] / f"{best['name']}.keras",
                out / "model.keras")
    shutil.copy(C.DATA_PROCESSED / "feature_config.json",
                out / "feature_config.json")
    shutil.copy(C.DATA_PROCESSED / "norm_stats.json", out / "norm_stats.json")
    joblib.dump(le, out / "label_encoder.joblib")

    card = {
        "project": "Speech Emotion Recognition (RAVDESS, speaker-independent)",
        "selected_model": best["name"],
        "architecture": best["model_key"],
        "feature_view": view,
        "selection_criterion": sel_info["rule"] + "; test set untouched",
        "selection_details": sel_info,
        "classes": le.classes_.tolist(),
        "n_classes": int(len(le.classes_)),
        "params": best["params"],
        "trained_on_augmented_data": best["augmented"],
        "class_weighting": best["class_weight"],
        "epochs_run": best["epochs_run"],
        "train_seconds": best["train_seconds"],
        "validation": {"accuracy": best["val_accuracy"],
                       "macro_f1": best["val_macro_f1"],
                       "weighted_f1": best["val_weighted_f1"]},
        "confidence_threshold": calib["threshold"],
        "confidence_calibration": {k: v for k, v in calib.items() if k != "sweep"},
        "split": {"train_actors": C.TRAIN_ACTORS, "val_actors": C.VAL_ACTORS,
                  "test_actors": C.TEST_ACTORS},
        "seed": C.SEED,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    (C.METRICS_DIR / "confidence_calibration.json").write_text(
        json.dumps(calib, indent=2))
    (C.METRICS_DIR / "validation_ranking.json").write_text(
        json.dumps(ranked, indent=2))
    print(f"\nBundle written to {out}")
    for f in sorted(out.iterdir()):
        print("  ", f.name, f"{f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
