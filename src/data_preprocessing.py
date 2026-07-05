"""
data_preprocessing.py
----------------------
Loads the credit card transactions dataset and applies:
  1. StandardScaler normalization (fit on train only, applied to test)
  2. Stratified train/test split (preserves the ~0.17% fraud rate in both splits)
  3. SMOTE oversampling of the minority (fraud) class on the TRAINING set only

Returns numpy arrays ready to feed into either the ANN or the RNN model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

REAL_FILENAME = "creditcard.csv"
SYNTHETIC_FILENAME = "creditcard_synthetic.csv"


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Load the real Kaggle file if present, else fall back to the synthetic one."""
    real_path = os.path.join(data_dir, REAL_FILENAME)
    synthetic_path = os.path.join(data_dir, SYNTHETIC_FILENAME)

    if os.path.exists(real_path):
        print(f"Loading real dataset from {real_path}")
        df = pd.read_csv(real_path)
    elif os.path.exists(synthetic_path):
        print(f"Real dataset not found. Loading synthetic dataset from {synthetic_path}")
        df = pd.read_csv(synthetic_path)
    else:
        raise FileNotFoundError(
            "No dataset found. Place 'creditcard.csv' (Kaggle Credit Card Fraud "
            "Detection dataset) in the data/ folder, or run "
            "src/generate_synthetic_data.py to create a synthetic one."
        )
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    smote_strategy: float = 0.5,
):
    """
    Returns X_train, X_test, y_train, y_test (all numpy arrays), already
    scaled, split, and (optionally) SMOTE-balanced on the training set.

    smote_strategy: desired ratio of minority:majority AFTER resampling
                    (0.5 means fraud samples become 50% of legit count;
                    tune this instead of full 1:1 to avoid over-diluting
                    the fraud signal with synthetic points).
    """
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # Stratified split preserves the fraud rate in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on train only to avoid data leakage; apply to both splits
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_smote:
        print(f"Pre-SMOTE class distribution (train): {np.bincount(y_train)}")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Post-SMOTE class distribution (train): {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = load_dataset(data_dir)
    print(df["Class"].value_counts())
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
