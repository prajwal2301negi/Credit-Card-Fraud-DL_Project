"""
rnn_model.py
------------
An RNN (stacked LSTM) branch of the project. Credit-card fraud datasets are
not naturally sequential (each row is one transaction), so to make an RNN
meaningful we treat each transaction's feature vector as a short sequence
of "steps" (reshape (n_features,) -> (timesteps, features_per_step)). This
lets the LSTM learn interactions across feature groups the way it would
learn temporal dependencies in a true time series.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def reshape_for_rnn(X: np.ndarray, timesteps: int = 4):
    """
    Reshape a (n_samples, n_features) matrix into (n_samples, timesteps,
    features_per_step) for the LSTM input. Pads with zeros if n_features
    isn't evenly divisible by timesteps.
    """
    n_samples, n_features = X.shape
    features_per_step = int(np.ceil(n_features / timesteps))
    pad_width = features_per_step * timesteps - n_features

    if pad_width > 0:
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")

    return X.reshape(n_samples, timesteps, features_per_step)


def build_rnn_model(timesteps: int, features_per_step: int, dropout_rate: float = 0.3,
                     learning_rate: float = 1e-3):
    model = models.Sequential(name="RNN_Fraud_Detector")
    model.add(layers.Input(shape=(timesteps, features_per_step)))

    # Stacked LSTM layers to learn sequential feature interactions
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.LSTM(32, return_sequences=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            "accuracy",
        ],
    )
    return model


if __name__ == "__main__":
    m = build_rnn_model(timesteps=4, features_per_step=8)
    m.summary()
