"""
Trains biased and debiased loan approval models.
Biased model: trained on raw biased data.
Debiased model: trained with sample reweighting to reduce gender bias.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/loan_data.csv"
MODEL_DIR = "model"


def load_data():
    df = pd.read_csv(DATA_PATH)
    features = ["age", "income", "credit_score", "employment_years", "gender"]
    X = df[features]
    y = df["approved"]
    sensitive = df["gender"]
    return X, y, sensitive, df


def compute_sample_weights(X, y, sensitive_col="gender"):
    """
    Reweighting: balance the dataset so each (gender, label) group
    has equal influence during training.
    """
    df_temp = X.copy()
    df_temp["label"] = y.values
    group_counts = df_temp.groupby([sensitive_col, "label"]).size()
    total = len(df_temp)
    n_groups = len(group_counts)
    weights = np.ones(len(df_temp))

    for (g, l), count in group_counts.items():
        mask = (df_temp[sensitive_col] == g) & (df_temp["label"] == l)
        weights[mask] = total / (n_groups * count)

    return weights


def train_models():
    X, y, sensitive, df = load_data()

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    # --- Biased Model ---
    biased_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    biased_pipeline.fit(X_train, y_train)

    # --- Debiased Model (Reweighting) ---
    sample_weights = compute_sample_weights(X_train, y_train, sensitive_col="gender")
    debiased_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    debiased_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # Save models and test data
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/biased_model.pkl", "wb") as f:
        pickle.dump(biased_pipeline, f)
    with open(f"{MODEL_DIR}/debiased_model.pkl", "wb") as f:
        pickle.dump(debiased_pipeline, f)

    test_data = pd.concat([X_test.reset_index(drop=True),
                           y_test.reset_index(drop=True),
                           s_test.reset_index(drop=True)], axis=1)
    test_data.to_csv(f"{MODEL_DIR}/test_data.csv", index=False)

    print("Models trained and saved.")
    return biased_pipeline, debiased_pipeline, X_test, y_test, s_test


if __name__ == "__main__":
    train_models()
