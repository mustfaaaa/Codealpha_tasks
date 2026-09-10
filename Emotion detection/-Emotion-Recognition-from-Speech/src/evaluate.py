"""PHASES 15-17, 19 -- evaluation on the untouched, speaker-independent test set.

Produces per-model metrics, confusion matrices, ROC / precision-recall curves,
training curves, a comparison table and an error-analysis CSV.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from train import load_norm, load_pack, normalize  # noqa: E402
from models import FEATURE_VIEW  # noqa: E402


def core_metrics(y_true, y_pred, y_prob, classes) -> dict:
    """Accuracy plus macro/weighted precision, recall, F1 and one-vs-rest AUC.

    Accuracy alone is not enough: with an 8-class problem where neutral has
    half the recordings of every other emotion, a model can post a decent
    accuracy while essentially never predicting neutral.  Macro-F1 exposes
    that because it averages the per-class F1 without weighting by support.
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        out["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except ValueError as exc:  # not all classes present
        out["roc_auc_ovr_macro"] = None
        out["roc_auc_note"] = str(exc)
    rep = classification_report(y_true, y_pred, target_names=classes,
                                output_dict=True, zero_division=0)
    out["per_class"] = {c: rep[c] for c in classes}
    return out


def plot_confusion(y_true, y_pred, classes, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{title} — counts")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"{title} — row-normalised (recall per class)")
    for ax in axes:
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(history: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — loss")
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — accuracy")
    if "val_macro_f1" in history:
        axes[2].plot(history["val_macro_f1"], color="darkgreen")
        best = int(np.argmax(history["val_macro_f1"]))
        axes[2].axvline(best, ls="--", color="crimson",
                        label=f"best epoch {best + 1}")
        axes[2].legend()
    axes[2].set_title(f"{title} — validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, classes, title: str, path: Path) -> None:
    yb = label_binarize(y_true, classes=range(len(classes)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        axes[0].plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
        pr, rc, _ = precision_recall_curve(yb[:, i], y_prob[:, i])
        ap = average_precision_score(yb[:, i], y_prob[:, i])
        axes[1].plot(rc, pr, label=f"{c} (AP={ap:.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                title=f"{title} — one-vs-rest ROC")
    axes[1].axhline(1 / len(classes), ls="--", color="k", lw=1, label="chance")
    axes[1].set(xlabel="recall", ylabel="precision",
                title=f"{title} — precision-recall")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_keras(model_key: str, name: str, x_test, y_test, classes,
                   test_pack) -> dict:
    path = C.MODELS_DIR / model_key / f"{name}.keras"
    model = tf.keras.models.load_model(path)
    t0 = time.time()
    prob = model.predict(x_test, verbose=0)
    infer_ms = (time.time() - t0) / len(x_test) * 1000
    pred = prob.argmax(axis=1)

    m = core_metrics(y_test, pred, prob, classes)
    m.update(name=name, model_key=model_key, params=int(model.count_params()),
             inference_ms_per_clip=round(float(infer_ms), 3))

    plot_confusion(y_test, pred, classes, name,
                   C.FIG_DIR / f"cm_{name}.png")
    plot_roc_pr(y_test, prob, classes, name,
                C.FIG_DIR / f"roc_pr_{name}.png")
    hist_p = C.MODELS_DIR / model_key / f"{name}_history.json"
    if hist_p.exists():
        plot_curves(json.loads(hist_p.read_text()), name,
                    C.FIG_DIR / f"curves_{name}.png")
    np.save(C.MODELS_DIR / model_key / f"{name}_test_prob.npy", prob)
    return m


def error_analysis(name: str, y_true, pred, prob, classes, test_pack) -> pd.DataFrame:
    manifest = pd.read_csv(C.DATA_PROCESSED / "manifest.csv")
    meta = manifest.set_index("filename")
    rows = []
    for i, fn in enumerate(test_pack["filename"]):
        if pred[i] == y_true[i]:
            continue
        r = meta.loc[fn]
        rows.append({
            "filename": fn,
            "true": classes[y_true[i]],
            "predicted": classes[pred[i]],
            "confidence": float(prob[i].max()),
            "true_class_prob": float(prob[i][y_true[i]]),
            "actor": int(r["actor"]),
            "gender": r["gender"],
            "intensity": r["intensity"],
            "duration": float(r["duration"]),
        })
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    df.to_csv(C.PRED_DIR / f"misclassified_{name}.csv", index=False)
    return df


def main() -> None:
    norm = load_norm()
    test = load_pack("test")
    tr = load_pack("train_clean")
    classes = sorted(set(tr["label"].tolist()))
    le = LabelEncoder().fit(classes)
    y_test = le.transform(test["label"])

    all_metrics: dict[str, dict] = {}

    # ---- classical baselines -------------------------------------------
    bdir = C.MODELS_DIR / "baseline"
    bres = json.loads((bdir / "baseline_results.json").read_text())
    x_stats = normalize(test["stats"], "stats", norm)
    for bname in bres["results"]:
        clf = joblib.load(bdir / f"{bname}.joblib")
        t0 = time.time()
        prob = clf.predict_proba(x_stats)
        infer_ms = (time.time() - t0) / len(x_stats) * 1000
        pred = prob.argmax(axis=1)
        m = core_metrics(y_test, pred, prob, classes)
        m.update(name=f"baseline_{bname}", model_key="baseline",
                 params=None, inference_ms_per_clip=round(float(infer_ms), 3))
        all_metrics[m["name"]] = m
        if bname == bres["best"]:
            plot_confusion(y_test, pred, classes, f"baseline_{bname}",
                           C.FIG_DIR / f"cm_baseline_{bname}.png")

    # ---- deep models ----------------------------------------------------
    for model_key in ("cnn", "lstm", "cnn_lstm"):
        d = C.MODELS_DIR / model_key
        if not d.exists():
            continue
        for res_file in sorted(d.glob("*_result.json")):
            name = json.loads(res_file.read_text())["name"]
            view = FEATURE_VIEW[model_key]
            x_test = normalize(test[view], view, norm)
            m = evaluate_keras(model_key, name, x_test, y_test, classes, test)
            all_metrics[name] = m
            prob = np.load(d / f"{name}_test_prob.npy")
            error_analysis(name, y_test, prob.argmax(axis=1), prob, classes, test)

    (C.METRICS_DIR / "test_metrics_all.json").write_text(
        json.dumps(all_metrics, indent=2))

    # ---- comparison table ----------------------------------------------
    rows = [{
        "Model": m["name"],
        "Accuracy": round(m["accuracy"], 4),
        "Precision(macro)": round(m["precision_macro"], 4),
        "Recall(macro)": round(m["recall_macro"], 4),
        "Macro F1": round(m["f1_macro"], 4),
        "Weighted F1": round(m["f1_weighted"], 4),
        "ROC-AUC(ovr)": round(m["roc_auc_ovr_macro"], 4) if m.get("roc_auc_ovr_macro") else None,
        "Params": m.get("params"),
        "ms/clip": m.get("inference_ms_per_clip"),
    } for m in all_metrics.values()]
    table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    table.to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = table.melt(id_vars="Model",
                         value_vars=["Accuracy", "Macro F1", "Weighted F1"],
                         var_name="metric", value_name="score")
    sns.barplot(data=plot_df, x="Model", y="score", hue="metric",
                palette="Set2", ax=ax)
    ax.set_title("Test-set performance (speaker-independent, unseen actors)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "10_model_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
