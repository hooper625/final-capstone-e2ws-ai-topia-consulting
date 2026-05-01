#!/usr/bin/env python3
"""
Model 4: NLP Complaint Routing — Prediction Script
====================================================
Loads the trained routing classifier and predicts the responsible agency
for each complaint in test_data/.

Artifacts expected in models/model4_nlp_classification/saved_model/:
  model4_routing_classifier_char_tfidf_SGD.pkl
  model4_routing_label_encoder.pkl

To run from project root:
    python -u models/model4_nlp_classification/predict.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE  = TEST_DATA_DIR / "model4_results.csv"

ROUTING_MODEL_FILE = MODEL_DIR / "model4_routing_classifier_char_tfidf_SGD.pkl"
ROUTING_LABEL_FILE = MODEL_DIR / "model4_routing_label_encoder.pkl"


class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer — must match the version used during training."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str).str.lower()
        extra = pd.DataFrame({
            "is_driveway": s.str.contains(r"\bdriveway\b", regex=True).astype(int),
            "is_parking":  s.str.contains(r"\bparking\b|\bparked\b|\bvehicle\b|\bcar\b", regex=True).astype(int),
            "is_blocked":  s.str.contains(r"\bblocked\b|\bblocking\b", regex=True).astype(int),
            "is_noise":    s.str.contains(r"\bbanging\b|\bpounding\b|\bloud\b|\bmusic\b", regex=True).astype(int),
        })
        return csr_matrix(extra.values)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model4_predict")
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


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_artifacts(logger: logging.Logger):
    for f in [ROUTING_MODEL_FILE, ROUTING_LABEL_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Missing artifact: {f}\nRun train.py first.")
    logger.info("Loading routing model from %s", MODEL_DIR)
    model = joblib.load(ROUTING_MODEL_FILE)
    label_encoder = joblib.load(ROUTING_LABEL_FILE)
    return model, label_encoder


def build_routing_text(df: pd.DataFrame) -> pd.Series:
    ct  = df.get("complaint_type", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    dsc = df.get("descriptor",     pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    res = df.get("resolution_description", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    return (ct + " | " + dsc + " | " + res).str.strip(" |")


def compute_confidence(model, X_text: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_text))
        return proba.max(axis=1).round(4).astype(float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_text))
        if scores.ndim == 1:
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.maximum(probs, 1.0 - probs).round(4).astype(float)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs.max(axis=1).round(4).astype(float)
    return np.ones(len(X_text), dtype=float)


def main():
    logger = setup_logging()
    model, label_encoder = load_artifacts(logger)

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

    routing_text = build_routing_text(df)

    pred_encoded = model.predict(routing_text)
    pred_labels  = label_encoder.inverse_transform(np.asarray(pred_encoded).astype(int))
    confidence   = compute_confidence(model, routing_text)

    results = pd.DataFrame({
        "id":              ids,
        "predicted_class": pred_labels,
        "confidence":      confidence,
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info("Predictions saved to %s", OUTPUT_FILE)
    print(results.head())


if __name__ == "__main__":
    main()
