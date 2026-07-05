"""
generate_synthetic_data.py
---------------------------
Generates a synthetic dataset that mimics the structure of the well-known
Kaggle "Credit Card Fraud Detection" dataset (284,807 transactions, 28
anonymized PCA features V1-V28, Time, Amount, Class), so the pipeline can
be run and tested end-to-end even without downloading the real file.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)


def generate(n_samples: int = 50_000, fraud_rate: float = 0.0017, n_features: int = 28):
    """Create a synthetic, class-imbalanced transactions dataset."""
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # Legitimate transactions: features drawn from a standard normal (like PCA components)
    legit_features = RNG.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    legit_amount = RNG.exponential(scale=60, size=n_legit)
    legit_time = RNG.uniform(0, 172_792, size=n_legit)  # ~2 days in seconds, like real dataset
    legit_labels = np.zeros(n_legit)

    # Fraudulent transactions: shifted mean / larger variance to be separable but noisy
    fraud_features = RNG.normal(loc=1.5, scale=2.2, size=(n_fraud, n_features))
    fraud_amount = RNG.exponential(scale=180, size=n_fraud)
    fraud_time = RNG.uniform(0, 172_792, size=n_fraud)
    fraud_labels = np.ones(n_fraud)

    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time = np.concatenate([legit_time, fraud_time])
    labels = np.concatenate([legit_labels, fraud_labels])

    col_names = [f"V{i+1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=col_names)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = labels.astype(int)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate()
    out_path = os.path.join(out_dir, "creditcard_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to {out_path}")
    print(df["Class"].value_counts(normalize=True))
