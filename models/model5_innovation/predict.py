#!/usr/bin/env python3
"""
Model 5: Resolution Outcome Predictor (NLP) — Prediction Script
================================================================
Loads the trained NLP artifacts and predicts complaint resolution outcome
(Resolved / Unresolved / Referred) for each row in the test CSV.

Artifacts expected in models/model5_innovation/saved_model/:
  outcome_clf.joblib, time_clf.joblib, tfidf.joblib,
  ord_enc.joblib, outcome_le.joblib, time_le.joblib, metrics.joblib

To run from project root:
    python -u models/model5_innovation/predict.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODEL_DIR     = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE   = TEST_DATA_DIR / "model5_results.csv"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model5_predict")
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
    required = [
        "outcome_clf.joblib", "tfidf.joblib", "ord_enc.joblib",
        "outcome_le.joblib", "metrics.joblib",
    ]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {MODEL_DIR}: {missing}\n"
            "Run train.py first to generate them."
        )
    logger.info("Loading artifacts from %s", MODEL_DIR)
    outcome_clf = joblib.load(MODEL_DIR / "outcome_clf.joblib")
    tfidf       = joblib.load(MODEL_DIR / "tfidf.joblib")
    ord_enc     = joblib.load(MODEL_DIR / "ord_enc.joblib")
    outcome_le  = joblib.load(MODEL_DIR / "outcome_le.joblib")
    metrics     = joblib.load(MODEL_DIR / "metrics.joblib")
    return outcome_clf, tfidf, ord_enc, outcome_le, metrics


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_features(df: pd.DataFrame, tfidf, ord_enc) -> object:
    input_text = (
        df["complaint_type"].fillna("") + " " + df["descriptor"].fillna("")
    ).apply(clean_text).values

    borough   = df["borough"].fillna("Unspecified") if "borough" in df.columns else pd.Series(["Unspecified"] * len(df))
    agency    = df["agency"].fillna("Unknown") if "agency" in df.columns else pd.Series(["Unknown"] * len(df))
    channel   = df["open_data_channel_type"].fillna("UNKNOWN") if "open_data_channel_type" in df.columns else pd.Series(["UNKNOWN"] * len(df))

    X_cat = ord_enc.transform(
        pd.DataFrame({
            "agency":                 agency.values,
            "borough":                borough.values,
            "open_data_channel_type": channel.values,
        }).values
    )

    hour        = pd.to_datetime(df.get("created_date", pd.Series([None] * len(df))), errors="coerce").dt.hour.fillna(12).astype(float).values
    day_of_week = pd.to_datetime(df.get("created_date", pd.Series([None] * len(df))), errors="coerce").dt.dayofweek.fillna(0).astype(float).values
    month       = pd.to_datetime(df.get("created_date", pd.Series([None] * len(df))), errors="coerce").dt.month.fillna(1).astype(float).values

    X_num    = np.column_stack([X_cat, hour, day_of_week, month])
    X_tfidf  = tfidf.transform(input_text)
    return hstack([X_tfidf, csr_matrix(X_num)])


def main():
    logger = setup_logging()
    outcome_clf, tfidf, ord_enc, outcome_le, metrics = load_artifacts(logger)

    metric_value = round(float(metrics.get("outcome_f1", metrics.get("outcome_f1_weighted", 0.0))), 4)
    logger.info("Model outcome F1: %.4f", metric_value)

    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name != OUTPUT_FILE.name and not p.name.startswith(".")
    ]
    if not candidates:
        raise FileNotFoundError(f"No test CSV found in {TEST_DATA_DIR}")

    preferred = ["urbanpulse_311_complaints_test.csv", "test.csv"]
    test_file = None
    for name in preferred:
        match = next((p for p in candidates if p.name.lower() == name.lower()), None)
        if match:
            test_file = match
            break
    if test_file is None:
        test_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    logger.info("Using test file: %s", test_file)
    df = pd.read_csv(test_file)
    logger.info("Loaded test data: %s", df.shape)

    id_col = next((c for c in df.columns if c.lower() in ("unique_key", "id")), None)
    ids = df[id_col].values if id_col else np.arange(1, len(df) + 1)

    X = build_features(df, tfidf, ord_enc)

    proba      = outcome_clf.predict_proba(X)
    pred_enc   = np.argmax(proba, axis=1)
    pred_labels = outcome_le.inverse_transform(pred_enc)
    confidence = np.round(proba.max(axis=1), 4)

    results = pd.DataFrame({
        "id":           ids,
        "prediction":   pred_labels,
        "confidence":   confidence,
        "metric_name":  "outcome_f1_weighted",
        "metric_value": metric_value,
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info("Predictions saved to %s", OUTPUT_FILE)
    print(results.head())


if __name__ == "__main__":
    main()
