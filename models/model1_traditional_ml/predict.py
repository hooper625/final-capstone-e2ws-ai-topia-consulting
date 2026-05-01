#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
=============================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model1_results.csv
"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH    = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE   = TEST_DATA_DIR / "model1_results.csv"

import numpy as np
import joblib
from pipelines.data_pipeline import accident_predict_features

def load_model():
    model     = joblib.load(MODEL_PATH / "model.joblib")
    scaler    = joblib.load(MODEL_PATH / "scaler.joblib")
    le        = joblib.load(MODEL_PATH / "label_encoder.joblib")
    features  = joblib.load(MODEL_PATH / "feature_columns.joblib")
    threshold_path = MODEL_PATH / "threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    return model, scaler, le, features, threshold

def predict(model, scaler, feature_cols, threshold, test_df):
    processed_df = accident_predict_features(test_df)
    processed_df = processed_df.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(processed_df)
    proba    = model.predict_proba(X_scaled)[:, 1]
    preds    = (proba >= threshold).astype(int)
    # Confidence = how far the probability is from the decision boundary
    confidence = np.where(proba >= threshold, proba, 1 - proba).round(4)
    return preds, confidence


def main():
    model, scaler, le, feature_cols, threshold = load_model()
    print(f"Decision threshold: {threshold:.2f}")

    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name != OUTPUT_FILE.name and not p.name.startswith(".")
    ]
    if not candidates:
        print(f"Error: No test CSV found in {TEST_DATA_DIR}")
        return

    preferred = ["city_traffic_accidents_test.csv", "city_traffic_accidents.csv"]
    test_file = None
    for name in preferred:
        match = next((p for p in candidates if p.name.lower() == name.lower()), None)
        if match:
            test_file = match
            break
    if test_file is None:
        test_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    print(f"Using test file: {test_file}")
    test_df = pd.read_csv(test_file)

    preds, confidence = predict(model, scaler, feature_cols, threshold, test_df)

    id_col = next((c for c in test_df.columns if c.lower() == 'id'), None)
    ids = test_df[id_col].values if id_col else range(1, len(test_df) + 1)

    pred_labels = np.vectorize(le.get)(preds) if isinstance(le, dict) else le.inverse_transform(preds)

    results = pd.DataFrame({
        'id':          ids,
        'prediction':  pred_labels,
        'probability': confidence,
        'confidence':  confidence,
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
