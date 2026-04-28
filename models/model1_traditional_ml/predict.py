#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
Usage: python predict.py
Output: test_data/model1_results.csv
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"

from pipelines.data_pipeline import load_raw_data, clean_data, drop_low_variance_columns
from pipelines.data_cleaning_accident_pipeline import accident_engineer_features


def load_model():
    model       = joblib.load(SAVED_MODEL_DIR / "model.joblib")
    scaler      = joblib.load(SAVED_MODEL_DIR / "scaler.joblib")
    le          = joblib.load(SAVED_MODEL_DIR / "label_encoder.joblib")
    feature_cols= joblib.load(SAVED_MODEL_DIR / "feature_columns.joblib")
    return model, scaler, le, feature_cols


def load_data():
    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name != OUTPUT_FILE.name and "model" not in p.name.lower()
    ]
    test_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"Test file: {test_file.name}")
    return pd.read_csv(test_file)


def preprocess_features(df):
    df = clean_data(df)
    df = accident_engineer_features(df)
    df = drop_low_variance_columns(df)
    df = df.dropna(axis=1)
    return df


def predict(model, test_data):
    proba = model.predict_proba(test_data)
    y_pred_encoded = model.predict(test_data)
    confidence = np.round(proba.max(axis=1), 6)
    return y_pred_encoded, confidence


def main():
    model, scaler, le, feature_cols = load_model()

    df = load_data()
    ids = df["ID"] if "ID" in df.columns else pd.Series(range(1, len(df) + 1))

    df = preprocess_features(df)

    # Align to the exact columns the model was trained on
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = df[col] if col in df.columns else 0

    X_scaled = scaler.transform(X)

    y_pred_encoded, confidence = predict(model, X_scaled)
    y_pred = le.inverse_transform(y_pred_encoded)

    results = pd.DataFrame({
        "id":         ids.values,
        "prediction": y_pred,
        "probability":confidence,
        "confidence": confidence,
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
