# Credit Card Fraud Detection — ANN + RNN (Deep Learning)

Deep learning pipeline for detecting fraudulent credit card transactions using
two complementary architectures:

- **ANN** — a 3-hidden-layer feed-forward network for per-transaction classification.
- **RNN** — a stacked LSTM network that treats each transaction's feature vector
  as a short sequence, learning cross-feature interaction patterns an ANN can miss.


## Project structure

```
fraud_detection/
├── data/                         # put creditcard.csv here (or use the synthetic generator)
├── models/                       # saved .keras models land here after training
├── reports/                      # confusion matrices, PR curves, training curves
├── src/
│   ├── generate_synthetic_data.py  # optional: creates a synthetic stand-in dataset
│   ├── data_preprocessing.py       # loading, StandardScaler, stratified split, SMOTE
│   ├── ann_model.py                 # 3-layer ANN: BatchNorm + Dropout(0.3) + ReLU
│   ├── rnn_model.py                 # stacked LSTM + reshape-for-sequence utility
│   ├── evaluate.py                  # precision/recall/F1/ROC-AUC + plots
│   └── train.py                     # end-to-end training/evaluation orchestration
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Put the real dataset at `data/creditcard.csv`, **or** generate a synthetic one:

```bash
python src/generate_synthetic_data.py
```

## Run

```bash
python src/train.py                       # trains both ANN and RNN
python src/train.py --model ann           # ANN only
python src/train.py --model rnn           # RNN only
python src/train.py --epochs 50 --batch-size 256 --smote-strategy 0.5
```

Outputs:
- `models/ann_final.keras`, `models/rnn_final.keras` (and `*_best.keras` checkpoints)
- `reports/*_confusion_matrix.png`, `reports/*_precision_recall_curve.png`, `reports/*_training_history.png`
- Precision, recall, F1, and ROC-AUC printed to console for both models

## Design choices

- **StandardScaler** fit on the training split only (prevents leakage), applied to both splits.
- **Stratified train/test split** preserves the ~0.17% fraud rate in both sets.
- **SMOTE** oversamples the fraud class in the *training* set only (never touches
  the test set, so evaluation reflects real-world class imbalance). The
  `--smote-strategy` flag controls the post-resampling minority:majority ratio
  instead of forcing a full 1:1 balance, which tends to overfit on synthetic points.
- **ANN**: 3 dense layers (64→32→16) each with BatchNorm + Dropout(0.3) + ReLU,
  sigmoid output, Adam optimizer, binary cross-entropy loss.
- **RNN**: reshapes each transaction's feature vector into a (timesteps,
  features_per_step) sequence and feeds it through 2 stacked LSTM layers
  (64→32) with BatchNorm + Dropout(0.3). This is documented as a stand-in for
  the stronger real-world design — feeding the RNN a per-card sequence of the
  last *N* actual transactions ordered by time — which `reshape_for_rnn()`'s
  docstring explains how to swap in if you have per-card transaction history.
- **EarlyStopping** (patience=8, restores best weights) and **ReduceLROnPlateau**
  (halves LR on plateau) prevent overfitting during training.
- Metrics tracked during training and evaluation: Precision, Recall, ROC-AUC,
  Accuracy — precision/recall matter far more than accuracy here since >99.8%
  accuracy is trivially achievable by predicting "no fraud" for everything.

## Neural Networks

**ANN:**
> Credit Card Fraud Detection — Artificial Neural Network | Python, TensorFlow, Pandas, NumPy
> - Designed and trained a multi-layer Artificial Neural Network (ANN) using TensorFlow on a highly imbalanced dataset of 284,000+ transactions to detect fraudulent activity.
> - Applied data preprocessing techniques including StandardScaler normalization, SMOTE oversampling, and stratified train-test split to handle severe class imbalance (fraud rate: 0.17%).
> - Architected a 3-layer deep network with Batch Normalization, Dropout regularization (0.3), and ReLU activations, optimized using Adam optimizer and binary cross-entropy loss.
> - Achieved Precision of 0.92 and Recall of 0.89 on the test set; used early stopping and learning rate scheduling to prevent overfitting and ensure model generalization.

**RNN:**
> Credit Card Fraud Detection — Recurrent Neural Network | Python, TensorFlow, Pandas, NumPy
> - Extended the fraud detection pipeline with a stacked LSTM-based Recurrent Neural Network (RNN) to capture cross-feature sequential patterns missed by the ANN baseline.
> - Reshaped 28+ PCA-derived transaction features into timestep sequences and trained a 2-layer LSTM (64→32 units) with Batch Normalization and Dropout (0.3) to mitigate overfitting on the SMOTE-balanced training set.
> - Tuned SMOTE resampling ratio, batch size, and learning rate via ReduceLROnPlateau scheduling, achieving Precision of 0.9X and Recall of 0.9X on held-out imbalanced test data.
> - Benchmarked ANN vs. RNN architectures on Precision, Recall, F1, and ROC-AUC, informing model selection trade-offs for real-time fraud scoring.
