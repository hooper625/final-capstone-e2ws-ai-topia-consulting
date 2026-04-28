#!/usr/bin/env python3
"""
Model 5: Innovation — Prediction Script
=========================================
Loads the trained resolution-time urgency model and generates predictions
for 311 complaint test data.

Usage:  python models/model5_innovation/predict.py
Output: test_data/model5_results.csv

Output columns: id, prediction, confidence, metric_name, metric_value
  - prediction:   high_risk | medium_risk | low_risk
  - confidence:   model's predicted probability for the winning class
  - metric_name:  "weighted_f1" (fixed)
  - metric_value: validation weighted F1 score from training (fixed per model)
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"

CATEGORICAL_FEATURES = ["complaint_type", "agency", "borough", "open_data_channel_type"]
NUMERIC_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_model():
    required = [
        SAVED_MODEL_DIR / "model.joblib",
        SAVED_MODEL_DIR / "ordinal_encoder.joblib",
        SAVED_MODEL_DIR / "label_encoder.joblib",
        SAVED_MODEL_DIR / "metrics.joblib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}\n"
            f"Run train.py first: python models/model5_innovation/train.py"
        )
    model = joblib.load(SAVED_MODEL_DIR / "model.joblib")
    enc = joblib.load(SAVED_MODEL_DIR / "ordinal_encoder.joblib")
    le = joblib.load(SAVED_MODEL_DIR / "label_encoder.joblib")
    metrics = joblib.load(SAVED_MODEL_DIR / "metrics.joblib")
    return model, enc, le, metrics


def detect_test_file():
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"test_data/ directory not found: {TEST_DATA_DIR}")

    exclude = {OUTPUT_FILE.name}
    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name not in exclude and not p.name.startswith(".")
        and "model" not in p.name.lower()
    ]

    preferred_names = [
        "urbanpulse_311_complaints_test.csv",
        "urbanpulse_311_complaints.csv",
        "test.csv",
        "test_data.csv",
    ]
    for name in preferred_names:
        for path in candidates:
            if path.name.lower() == name.lower():
                return path

    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(f"No test CSV found in {TEST_DATA_DIR}")


def extract_features(df):
    """Apply same feature engineering as train.py."""
    df = df.copy()

    date_col = next(
        (c for c in ["created_date", "Created_Date", "date"] if c in df.columns),
        None,
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["hour"] = df[date_col].dt.hour.fillna(12).astype(int)
        df["day_of_week"] = df[date_col].dt.dayofweek.fillna(0).astype(int)
        df["month"] = df[date_col].dt.month.fillna(1).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    else:
        df["hour"] = 12
        df["day_of_week"] = 0
        df["month"] = 1
        df["is_weekend"] = 0

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            df[col] = "Unknown"

    return df


def get_ids(df):
    for col in ["unique_key", "id", "ID", "Unique_Key"]:
        if col in df.columns:
            return df[col]
    return pd.Series(np.arange(1, len(df) + 1), name="id")


def predict(model, enc, le, metrics, test_df):
    test_df = extract_features(test_df)

    X_cat = enc.transform(test_df[CATEGORICAL_FEATURES])
    X_num = test_df[NUMERIC_FEATURES].values
    X_encoded = np.hstack([X_cat, X_num])

    y_pred_encoded = model.predict(X_encoded)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    proba = model.predict_proba(X_encoded)
    confidence = np.round(proba.max(axis=1), 6)

    ids = get_ids(test_df)

    return pd.DataFrame({
        "id": ids.values,
        "prediction": y_pred_labels,
        "confidence": confidence,
        "metric_name": "weighted_f1",
        "metric_value": metrics["weighted_f1"],
    })


def main():
    print("Loading model artifacts...")
    model, enc, le, metrics = load_model()
    print(f"Model loaded | Validation weighted F1: {metrics['weighted_f1']}")

    test_file = detect_test_file()
    print(f"Detected test file: {test_file.name}")
    test_df = pd.read_csv(test_file)
    print(f"Test data: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

    results = predict(model, enc, le, metrics, test_df)

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions saved to {OUTPUT_FILE}")
    print(f"Output: {len(results):,} rows")
    print(results.head())


if __name__ == "__main__":
    main()
