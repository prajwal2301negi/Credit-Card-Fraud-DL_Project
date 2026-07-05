"""
evaluate.py
-----------
Shared evaluation utilities for both the ANN and RNN models:
precision, recall, F1, ROC-AUC, confusion matrix, and plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5, out_dir: str = None):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n===== {model_name} Evaluation =====")
    for k, v in metrics.items():
        print(f"{k.capitalize():10s}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _plot_confusion_matrix(cm, model_name, out_dir)
        _plot_precision_recall_curve(y_test, y_prob, model_name, out_dir)

    return metrics, cm


def _plot_confusion_matrix(cm, model_name, out_dir):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {path}")


def _plot_precision_recall_curve(y_test, y_prob, model_name, out_dir):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name.lower()}_precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved precision-recall curve to {path}")


def plot_training_history(history, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics_to_plot = ["loss", "precision", "recall", "auc"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    for ax, metric in zip(axes, metrics_to_plot):
        if metric in history.history:
            ax.plot(history.history[metric], label="train")
            ax.plot(history.history.get(f"val_{metric}", []), label="val")
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.legend()
    fig.suptitle(f"{model_name} Training History")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name.lower()}_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {path}")
