#!/usr/bin/env python3
"""
Model 5: Innovation — 311 Complaint Resolution Time Urgency Predictor
=======================================================================
Predicts whether a 311 complaint will take a long time to resolve
(high_risk), moderate time (medium_risk), or short time (low_risk).

Value proposition: City agencies receive ~430K+ service complaints. This
model predicts resolution time up front, enabling smarter dispatch priority
and proactive SLA management — shifting from reactive to predictive operations.

Success metric: Weighted F1 (handles class imbalance across urgency tiers)

Cost-benefit: If flagging 15% of complaints as high_risk prevents even 10%
of SLA breaches, that's ~6,500 fewer late resolutions per year — reducing
resident escalations, council complaints, and staff overtime.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"

# Risk tier thresholds (hours)
LOW_RISK_MAX_HOURS = 24    # resolved same day
HIGH_RISK_MIN_HOURS = 120  # more than 5 days

CATEGORICAL_FEATURES = ["complaint_type", "agency", "borough", "open_data_channel_type"]
NUMERIC_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MAX_TRAIN_ROWS = 200_000  # cap for training speed; stratified sample


def load_data():
    path = RAW_DATA_DIR / "urbanpulse_311_complaints.csv"
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=["created_date", "closed_date"])
    print(f"Loaded {len(df):,} rows")
    return df


def create_target(df):
    """Compute resolution hours and map to risk label."""
    df = df.dropna(subset=["closed_date"]).copy()
    df["resolution_hours"] = (
        (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    )
    df = df[df["resolution_hours"] > 0].copy()

    def _label(hours):
        if hours < LOW_RISK_MAX_HOURS:
            return "low_risk"
        elif hours <= HIGH_RISK_MIN_HOURS:
            return "medium_risk"
        return "high_risk"

    df["risk_label"] = df["resolution_hours"].apply(_label)
    return df


def extract_features(df):
    """Extract temporal and categorical features."""
    df = df.copy()
    df["hour"] = df["created_date"].dt.hour
    df["day_of_week"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df


def preprocess(df):
    df = create_target(df)
    df = extract_features(df)
    X = df[ALL_FEATURES].copy()
    y = df["risk_label"]
    return X, y


def train_model(X_train, y_train):
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X_cat = enc.fit_transform(X_train[CATEGORICAL_FEATURES])
    X_num = X_train[NUMERIC_FEATURES].values
    X_encoded = np.hstack([X_cat, X_num])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_encoded, y_encoded)
    return model, enc, le


def evaluate_model(model, enc, le, X_val, y_val):
    X_cat = enc.transform(X_val[CATEGORICAL_FEATURES])
    X_num = X_val[NUMERIC_FEATURES].values
    X_encoded = np.hstack([X_cat, X_num])

    y_true = le.transform(y_val)
    y_pred = model.predict(X_encoded)

    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print("\n--- Model Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    # Baseline: predict majority class
    majority = np.bincount(y_true).argmax()
    baseline_f1 = f1_score(y_true, np.full_like(y_true, majority), average="weighted")
    print(f"Baseline (majority class) F1: {baseline_f1:.4f}")
    print(f"Improvement over baseline: +{weighted_f1 - baseline_f1:.4f}")

    return weighted_f1


def save_model(model, enc, le, weighted_f1):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(enc, SAVED_MODEL_DIR / "ordinal_encoder.joblib")
    joblib.dump(le, SAVED_MODEL_DIR / "label_encoder.joblib")
    joblib.dump(
        {"weighted_f1": round(float(weighted_f1), 4)},
        SAVED_MODEL_DIR / "metrics.joblib",
    )
    print(f"\nModel artifacts saved to {SAVED_MODEL_DIR}")


def main():
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)
    print(f"After filtering: {len(X):,} rows")
    print(f"Class distribution:\n{y.value_counts()}\n")

    # Stratified sample to cap training time
    if len(X) > MAX_TRAIN_ROWS:
        print(f"Sampling {MAX_TRAIN_ROWS:,} rows (stratified)...")
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=MAX_TRAIN_ROWS,
            stratify=y,
            random_state=42,
        )
        X, y = X_sample, y_sample
        print(f"Sample class distribution:\n{y.value_counts()}\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    print("\nTraining Gradient Boosting model...")
    model, enc, le = train_model(X_train, y_train)

    weighted_f1 = evaluate_model(model, enc, le, X_val, y_val)

    save_model(model, enc, le, weighted_f1)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
