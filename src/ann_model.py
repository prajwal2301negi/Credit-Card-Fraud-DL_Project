"""
ann_model.py
------------
A 3-hidden-layer Artificial Neural Network (ANN) for binary fraud
classification, matching the resume description:
  - Batch Normalization after each dense layer
  - Dropout(0.3) regularization
  - ReLU activations
  - Adam optimizer, binary cross-entropy loss
  - Sigmoid output for binary probability
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_ann_model(input_dim: int, dropout_rate: float = 0.3, learning_rate: float = 1e-3):
    model = models.Sequential(name="ANN_Fraud_Detector")
    model.add(layers.Input(shape=(input_dim,)))

    # Layer 1
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # Layer 2
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # Layer 3
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # Output layer
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
    m = build_ann_model(input_dim=29)
    m.summary()
