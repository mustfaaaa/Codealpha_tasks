"""Keras model definitions.

Three deep architectures, each justified by what it can express:

CNN (2-D)
    Treats the (n_mfcc x n_frames) map like an image and learns local
    time-frequency patterns with weight sharing.  Good at texture-like cues:
    harmonic spacing, spectral tilt, burst shapes.  It has no explicit notion
    of order beyond its receptive field.

BiLSTM
    Consumes one 120-dim vector per 32 ms frame and carries a state across
    time, so it can model how prosody *evolves* -- a rising then falling pitch
    contour, a slow decay in energy.  Bidirectional because the whole
    utterance is available offline, so the end of the sentence may inform the
    interpretation of its beginning.

CNN-LSTM
    Convolutions first compress each short time slice into a robust local
    descriptor, then the LSTM models the sequence of those descriptors.  This
    is the natural combination: local acoustic pattern extraction followed by
    long-range temporal integration, with far fewer recurrent steps than
    feeding raw frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, regularizers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def build_cnn(input_shape=C.CNN_INPUT_SHAPE, n_classes: int = 8,
              filters=(32, 64, 128), dropout: float = 0.3,
              dense_units: int = 128, l2: float = 1e-4,
              bn_momentum: float = C.BN_MOMENTUM) -> tf.keras.Model:
    """Conv2D -> BN -> ReLU -> MaxPool -> Dropout stack, then GAP + Dense.

    ``bn_momentum`` is deliberately below the Keras default of 0.99.  This
    dataset yields only ~30-90 gradient steps per epoch, and at 0.99 the
    BatchNormalization moving mean/variance lag far behind the true activation
    statistics.  Training then looks healthy (batch statistics are used) while
    validation collapses (moving statistics are used) -- which is exactly what
    we measured before fixing it.
    """
    inp = layers.Input(shape=input_shape, name="mfcc_stack")
    x = inp
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same",
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(f, 3, padding="same",
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
    # Global average pooling instead of Flatten: far fewer parameters, which
    # matters a lot with ~1000 training utterances.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout + 0.1)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="CNN2D")


def build_lstm(input_shape=C.SEQ_INPUT_SHAPE, n_classes: int = 8,
               units=(128, 64), dropout: float = 0.3,
               dense_units: int = 64, bidirectional: bool = True) -> tf.keras.Model:
    """Stacked (Bi)LSTM over per-frame MFCC+delta vectors."""
    inp = layers.Input(shape=input_shape, name="mfcc_sequence")
    x = layers.Masking(mask_value=0.0)(inp)
    for i, u in enumerate(units):
        last = i == len(units) - 1
        rnn = layers.LSTM(u, return_sequences=not last)
        x = layers.Bidirectional(rnn)(x) if bidirectional else rnn(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="BiLSTM")


def build_cnn_lstm(input_shape=C.CNN_INPUT_SHAPE, n_classes: int = 8,
                   filters=(32, 64, 128), lstm_units: int = 128,
                   dropout: float = 0.3, dense_units: int = 64,
                   bn_momentum: float = C.BN_MOMENTUM) -> tf.keras.Model:
    """CNN feature extractor along frequency, then BiLSTM along time.

    Pooling is applied on the frequency axis only (pool (2, 1)) so the time
    resolution reaching the recurrent layer stays intact.
    """
    inp = layers.Input(shape=input_shape, name="mfcc_stack")
    x = inp
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 1))(x)   # squeeze frequency, keep time
        x = layers.Dropout(dropout)(x)
    # (batch, freq', time, ch) -> (batch, time, freq'*ch)
    x = layers.Permute((2, 1, 3))(x)
    shp = x.shape
    x = layers.Reshape((shp[1], shp[2] * shp[3]))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="CNN_LSTM")


BUILDERS = {"cnn": build_cnn, "lstm": build_lstm, "cnn_lstm": build_cnn_lstm}
FEATURE_VIEW = {"cnn": "cnn", "lstm": "seq", "cnn_lstm": "cnn"}
