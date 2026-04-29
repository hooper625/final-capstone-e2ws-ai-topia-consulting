#!/usr/bin/env python3
"""
Model 2: Traffic Accident Severity — DNN Prediction Script
===========================================================
Loads saved artifacts and predicts severity class (1–4) for new data.

Artifacts expected in models/model2_deep_learning/saved_model/:
  model.keras, scaler.joblib, label_encoder.joblib,
  feature_columns.joblib, metrics.joblib

To run from project root:
    python -u models/model2_deep_learning/predict.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT    = Path.cwd()
MODEL_DIR       = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
TEST_DATA_DIR   = PROJECT_ROOT / "test_data"
OUTPUT_FILE     = TEST_DATA_DIR / "model2_dnn_results.csv"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model2_predict")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_artifacts(logger: logging.Logger):
    required = ["model.keras", "scaler.joblib", "label_encoder.joblib", "feature_columns.joblib"]
    missing  = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {MODEL_DIR}: {missing}\n"
            "Run train.py first to generate them."
        )
    logger.info("Loading model from %s", MODEL_DIR)
    model          = tf.keras.models.load_model(MODEL_DIR / "model.keras")
    scaler         = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder  = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_cols   = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, scaler, label_encoder, feature_cols


def apply_thresholds(proba: np.ndarray) -> np.ndarray:
    """Same custom thresholds used during training to boost minority-class recall."""
    preds = []
    for row in proba:
        if row[0] >= 0.30:
            preds.append(0)
        elif row[3] >= 0.20:
            preds.append(3)
        else:
            preds.append(int(np.argmax(row)))
    return np.array(preds)


def preprocess(df: pd.DataFrame, scaler, feature_cols: list) -> np.ndarray:
    X = df.select_dtypes(include=[np.number]).copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    return scaler.transform(X)


def predict(df: pd.DataFrame, model, scaler, label_encoder, feature_cols) -> pd.DataFrame:
    X_scaled      = preprocess(df, scaler, feature_cols)
    proba         = model.predict(X_scaled)
    pred_encoded  = apply_thresholds(proba)
    pred_labels   = label_encoder.inverse_transform(pred_encoded)
    confidence    = proba.max(axis=1)
    return pd.DataFrame({
        "predicted_severity": pred_labels,
        "confidence":         np.round(confidence, 4),
    })


def main():
    logger = setup_logging()
    model, scaler, label_encoder, feature_cols = load_artifacts(logger)

    candidates = [p for p in TEST_DATA_DIR.glob("*.csv") if p.name != OUTPUT_FILE.name]
    if not candidates:
        raise FileNotFoundError(f"No test CSV found in {TEST_DATA_DIR}")
    test_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logger.info("Using test file: %s", test_file)

    df = pd.read_csv(test_file)
    logger.info("Loaded test data: %s", df.shape)

    results = predict(df, model, scaler, label_encoder, feature_cols)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df[["Severity"]].reset_index(drop=True) if "Severity" in df.columns
               else pd.DataFrame(), results], axis=1).to_csv(OUTPUT_FILE, index=False)
    logger.info("Predictions saved to %s", OUTPUT_FILE)
    print(results.head())


if __name__ == "__main__":
    main()
