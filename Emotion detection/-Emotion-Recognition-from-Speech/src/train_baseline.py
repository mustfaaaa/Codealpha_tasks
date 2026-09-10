"""PHASE 9 -- classical machine-learning baselines.

Purpose: establish what a non-deep model achieves on the same speaker-
independent split, so the deep networks have to *earn* their complexity.
The baselines consume the fixed-length statistical descriptor (means and
standard deviations of MFCC/delta/delta2 plus low-level spectral features),
which throws away fine temporal structure -- exactly the information the CNN
and LSTM are supposed to exploit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from train import load_norm, load_pack, normalize  # noqa: E402


def main() -> None:
    norm = load_norm()
    tr = load_pack("train_aug")     # baselines get the same augmented pool
    va = load_pack("val")

    le = LabelEncoder().fit(sorted(set(tr["label"].tolist())))
    x_tr = normalize(tr["stats"], "stats", norm)
    x_va = normalize(va["stats"], "stats", norm)
    y_tr, y_va = le.transform(tr["label"]), le.transform(va["label"])

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, C=1.0,
                                     class_weight="balanced",
                                     random_state=C.SEED),
        "svm_rbf": SVC(C=10.0, gamma="scale", kernel="rbf", probability=True,
                       class_weight="balanced", random_state=C.SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced", random_state=C.SEED),
    }

    outdir = C.MODELS_DIR / "baseline"
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(x_tr, y_tr)
        fit_s = time.time() - t0
        pred = clf.predict(x_va)
        results[name] = {
            "name": name,
            "val_accuracy": float(accuracy_score(y_va, pred)),
            "val_macro_f1": float(f1_score(y_va, pred, average="macro")),
            "val_weighted_f1": float(f1_score(y_va, pred, average="weighted")),
            "fit_seconds": round(fit_s, 1),
        }
        joblib.dump(clf, outdir / f"{name}.joblib")
        print(name, json.dumps(results[name]))

    best = max(results, key=lambda k: results[k]["val_macro_f1"])
    print(f"best baseline on validation macro-F1: {best}")
    payload = {"results": results, "best": best,
               "classes": le.classes_.tolist()}
    (outdir / "baseline_results.json").write_text(json.dumps(payload, indent=2))
    joblib.dump(le, outdir / "label_encoder.joblib")
    np.save(outdir / "val_prob_best.npy",
            candidates[best].predict_proba(x_va))


if __name__ == "__main__":
    main()
