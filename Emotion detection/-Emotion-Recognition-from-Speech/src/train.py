"""Train a deep model (cnn | lstm | cnn_lstm) on the RAVDESS features.

Usage:
    python src/train.py --model cnn --aug
    python src/train.py --model lstm --no-aug --tag noaug
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from models import BUILDERS, FEATURE_VIEW  # noqa: E402


def set_seeds(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_pack(name: str) -> dict:
    z = np.load(C.DATA_PROCESSED / f"features_{name}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_norm() -> dict:
    return json.loads((C.DATA_PROCESSED / "norm_stats.json").read_text())


def normalize(x: np.ndarray, view: str, norm: dict) -> np.ndarray:
    """Standardise with statistics computed on the training split only."""
    if view == "cnn":
        m = np.asarray(norm["cnn_mean"], dtype=np.float32)
        s = np.asarray(norm["cnn_std"], dtype=np.float32)
        return ((x - m) / s).astype(np.float32)
    if view == "seq":
        m = np.asarray(norm["seq_mean"], dtype=np.float32)
        return ((x - m) / norm["seq_std"]).astype(np.float32)
    m = np.asarray(norm["stats_mean"], dtype=np.float32)
    s = np.asarray(norm["stats_std"], dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


class MacroF1(tf.keras.callbacks.Callback):
    """Log validation macro-F1 each epoch so callbacks can monitor it.

    Macro-F1 is the right early-stopping signal here: it weights every emotion
    equally, so the model cannot 'improve' by getting better at the majority
    classes while abandoning neutral.
    """

    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val, self.y_val = x_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        p = self.model.predict(self.x_val, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = float(f1_score(self.y_val, p, average="macro"))


def build_callbacks(ckpt: Path, x_val, y_val) -> list:
    return [
        MacroF1(x_val, y_val),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_macro_f1", mode="max",
            save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=C.EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]


def train_model(model_key: str, use_aug: bool = True, tag: str | None = None,
                lr: float = C.LR, batch_size: int = C.BATCH_SIZE,
                epochs: int = C.MAX_EPOCHS, use_class_weight: bool = True,
                build_kwargs: dict | None = None, quiet: bool = False) -> dict:
    set_seeds()
    view = FEATURE_VIEW[model_key]
    norm = load_norm()

    train_pack = load_pack("train_aug" if use_aug else "train_clean")
    val_pack = load_pack("val")

    le = LabelEncoder().fit(sorted(set(train_pack["label"].tolist())))
    y_tr = le.transform(train_pack["label"])
    y_va = le.transform(val_pack["label"])

    x_tr = normalize(train_pack[view], view, norm)
    x_va = normalize(val_pack[view], view, norm)

    # Class weights counter the neutral class having half as many recordings.
    if use_class_weight:
        counts = np.bincount(y_tr, minlength=len(le.classes_))
        cw = {i: float(len(y_tr) / (len(le.classes_) * c)) if c else 1.0
              for i, c in enumerate(counts)}
    else:
        cw = None

    builder = BUILDERS[model_key]
    model = builder(n_classes=len(le.classes_), **(build_kwargs or {}))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    name = tag or (f"{model_key}_aug" if use_aug else f"{model_key}_noaug")
    outdir = C.MODELS_DIR / model_key
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{name}.keras"

    t0 = time.time()
    hist = model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va),
        epochs=epochs, batch_size=batch_size, class_weight=cw,
        callbacks=build_callbacks(ckpt, x_va, y_va),
        verbose=0 if quiet else 2,
    )
    train_time = time.time() - t0

    val_prob = model.predict(x_va, verbose=0)
    val_pred = val_prob.argmax(axis=1)
    result = {
        "name": name,
        "model_key": model_key,
        "augmented": bool(use_aug),
        "class_weight": bool(use_class_weight),
        "lr": lr,
        "batch_size": batch_size,
        "build_kwargs": build_kwargs or {},
        "params": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "train_seconds": round(train_time, 1),
        "val_accuracy": float((val_pred == y_va).mean()),
        "val_macro_f1": float(f1_score(y_va, val_pred, average="macro")),
        "val_weighted_f1": float(f1_score(y_va, val_pred, average="weighted")),
        "classes": le.classes_.tolist(),
        "checkpoint": str(ckpt),
    }

    model.save(ckpt)  # weights already restored to the best epoch
    (outdir / f"{name}_history.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}))
    (outdir / f"{name}_result.json").write_text(json.dumps(result, indent=2))
    np.save(outdir / f"{name}_val_prob.npy", val_prob)
    print(json.dumps({k: result[k] for k in
                      ("name", "params", "epochs_run", "train_seconds",
                       "val_accuracy", "val_macro_f1")}, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BUILDERS))
    ap.add_argument("--aug", dest="aug", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    a = ap.parse_args()
    train_model(a.model, a.aug, a.tag, a.lr, a.batch_size, a.epochs)
