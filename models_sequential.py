"""
models_sequential.py
---------------------
LSTM, GRU, and a small Transformer for sequence-based forecasting.
These need data reshaped into (samples, timesteps, features) windows,
unlike the tabular models which take flat rows.

Improvements over a naive setup:
- More capacity (units) so the models can actually learn the pattern
  instead of underfitting.
- EarlyStopping with restore_best_weights, so we can set a generous
  epoch budget without wasting time or overfitting -- training stops
  automatically once validation loss stops improving.

Usage:
    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=30)
    model = build_lstm(input_shape=(30, num_features))
    model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
              callbacks=get_callbacks(), verbose=0)
    preds = model.predict(X_test_seq)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def make_sequences(X: np.ndarray, y: np.ndarray, window: int = 30):
    """
    Convert flat feature/target arrays into overlapping sequences.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    window : int
        Number of past days used to predict the next value.

    Returns
    -------
    X_seq : np.ndarray, shape (n_samples - window, window, n_features)
    y_seq : np.ndarray, shape (n_samples - window,)
        y_seq[i] corresponds to the target right after the i-th window.
    """
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def get_callbacks(patience=10):
    """Stop training once validation loss stops improving, and keep the
    best-performing weights rather than whatever the last epoch produced."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]


def build_lstm(input_shape, units=128, dropout=0.3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 4),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def build_gru(input_shape, units=128, dropout=0.3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.GRU(units // 2, return_sequences=True),
        layers.Dropout(dropout),
        layers.GRU(units // 4),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def build_transformer(input_shape, head_size=64, num_heads=4, ff_dim=128, dropout=0.2, num_blocks=2):
    """
    A compact Transformer encoder for time-series regression.
    Kept intentionally lightweight -- daily stock prediction rarely
    benefits from a large Transformer on only a few thousand rows;
    a small model trains faster and overfits less.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn = layers.Dropout(dropout)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(input_shape[-1])(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


SEQUENTIAL_MODEL_BUILDERS = {
    "LSTM": build_lstm,
    "GRU": build_gru,
    "Transformer": build_transformer,
}
