"""
train.py
--------
End-to-end training pipeline for both the ANN and RNN fraud detectors.

Usage:
    python src/train.py
    python src/train.py --model ann
    python src/train.py --model rnn
    python src/train.py --epochs 30 --batch-size 256
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data_preprocessing import load_dataset, preprocess
from ann_model import build_ann_model
from rnn_model import build_rnn_model, reshape_for_rnn
from evaluate import evaluate_model, plot_training_history

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def get_callbacks(model_name: str):
    return [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{model_name.lower()}_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=0,
        ),
    ]


def train_ann(X_train, X_test, y_train, y_test, epochs, batch_size):
    print("\n================ Training ANN ================")
    model = build_ann_model(input_dim=X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("ANN"),
        class_weight=None,  # SMOTE already balances the training set
        verbose=2,
    )

    plot_training_history(history, "ANN", REPORTS_DIR)
    evaluate_model(model, X_test, y_test, "ANN", out_dir=REPORTS_DIR)

    model.save(os.path.join(MODELS_DIR, "ann_final.keras"))
    return model, history


def train_rnn(X_train, X_test, y_train, y_test, epochs, batch_size, timesteps=4):
    print("\n================ Training RNN ================")
    X_train_seq = reshape_for_rnn(X_train, timesteps=timesteps)
    X_test_seq = reshape_for_rnn(X_test, timesteps=timesteps)
    features_per_step = X_train_seq.shape[2]

    model = build_rnn_model(timesteps=timesteps, features_per_step=features_per_step)
    model.summary()

    history = model.fit(
        X_train_seq, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("RNN"),
        verbose=2,
    )

    plot_training_history(history, "RNN", REPORTS_DIR)
    evaluate_model(model, X_test_seq, y_test, "RNN", out_dir=REPORTS_DIR)

    model.save(os.path.join(MODELS_DIR, "rnn_final.keras"))
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train ANN and/or RNN fraud detectors")
    parser.add_argument("--model", choices=["ann", "rnn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smote-strategy", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_dataset(DATA_DIR)
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, apply_smote=True, smote_strategy=args.smote_strategy
    )

    if args.model in ("ann", "both"):
        train_ann(X_train, X_test, y_train, y_test, args.epochs, args.batch_size)

    if args.model in ("rnn", "both"):
        train_rnn(X_train, X_test, y_train, y_test, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
