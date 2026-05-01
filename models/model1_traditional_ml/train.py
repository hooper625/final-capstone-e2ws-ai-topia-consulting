#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
===========================================
XGBoost binary classifier: High Risk (Severity 3+4) vs Standard Risk (1+2).
Includes threshold optimisation so the saved model maximises weighted F1.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier

PROCESSED_DATA  = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")

from pipelines.data_pipeline import (
    load_raw_data, clean_data, save_processed_data, drop_low_variance_columns,
    get_data_and_process_target, split_data, scale_features, accident_predict_features,
)
from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
from pipelines.Classification_pipelines import plot_feature_importance

UI_FEATURES = [
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
    'duration_min',
    'Distance(mi)', 'n_road_features', 'has_traffic_control',
    'is_freezing', 'low_visibility_severity', 'has_precipitation',
    'weather_cluster_clear', 'weather_cluster_cloudy',
    'weather_cluster_low_visibility', 'weather_cluster_rain',
    'weather_cluster_snow_ice', 'DangerousScore',
]


def load_data():
    return load_raw_data("city_traffic_accidents.csv")


def preprocess_features(df):
    TARGET = 'Severity'
    processed_path = PROCESSED_DATA / "city_traffic_processed.csv"

    # Regenerate if duration_min is missing from a prior cached version
    if processed_path.exists():
        import pandas as pd
        probe = pd.read_csv(processed_path, nrows=1)
        if 'duration_min' not in probe.columns:
            processed_path.unlink()
            print("Regenerating processed data with improved features...")

    if not processed_path.exists():
        df = clean_data(df)
        df = accident_engineer_features(df)
        df = drop_low_variance_columns(df)
        df = df.dropna(axis=1)
        available = [c for c in UI_FEATURES if c in df.columns]
        df = df[[TARGET] + available]
        save_processed_data(df, "city_traffic_processed.csv")
    else:
        print("Processed data found — loading from cache.")

    df, target_stats = get_data_and_process_target("city_traffic_processed.csv", target_column=TARGET)
    if target_stats:
        print(f"\nReady to process models for {TARGET}...")

    X = df.drop(columns=[TARGET]).copy()
    y = (df[TARGET] >= 3).astype(int)
    print(f"  Standard Risk (0): {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"  High Risk     (1): {(y==1).sum():,} ({(y==1).mean():.1%})")

    label_map = {0: "Standard Risk", 1: "High Risk"}

    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
    X_train_scaled, X_val_scaled, scaler, features = scale_features(X_train, X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler, label_map, features


def train_model(X_train, y_train, X_val, y_val):
    if len(X_train) > 80_000:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=80_000, random_state=42, stratify=y_train
        )
        print(f"Subsampled to {len(y_train):,} rows for training.")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    # Use half the natural ratio — improves precision without sacrificing too much recall
    scale_pw = (neg / pos) * 0.5
    print(f"  Class ratio neg/pos={neg/pos:.2f}  →  scale_pos_weight={scale_pw:.2f}")

    mdl = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        n_jobs=-1,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
    )
    mdl.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"  Best iteration: {mdl.best_iteration}")
    return mdl


def find_best_threshold(model, X_val, y_val):
    from sklearn.metrics import f1_score
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_val, preds, average='weighted')
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"  Optimal threshold: {best_thresh:.2f}  (weighted F1={best_f1:.4f})")
    return float(best_thresh)


def evaluate_model(model, X_val, y_val, threshold):
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    proba  = model.predict_proba(X_val)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print("\n--- Model Evaluation ---")
    print(f"Threshold     : {threshold:.2f}")
    print(f"Test Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"Test F1 (wtd) : {f1_score(y_val, y_pred, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred))
    return {"accuracy": accuracy_score(y_val, y_pred),
            "f1_weighted": f1_score(y_val, y_pred, average='weighted')}


def explain_model(model, X_val, y_val):
    print("\n--- Feature Importance Analysis ---")
    plot_feature_importance(model, X_val, y_val, "XGBoost")


def save_model(model, scaler, le, feature_cols, threshold):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(scaler,       SAVED_MODEL_DIR / "scaler.joblib")
    joblib.dump(le,           SAVED_MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_columns.joblib")
    joblib.dump(threshold,    SAVED_MODEL_DIR / "threshold.joblib")
    print(f"Artifacts saved to {SAVED_MODEL_DIR}")


def main():
    df = load_data()
    X_train, X_val, y_train, y_val, scaler, le, feature_cols = preprocess_features(df)
    model     = train_model(X_train, y_train, X_val, y_val)
    threshold = find_best_threshold(model, X_val, y_val)
    save_model(model, scaler, le, feature_cols, threshold)
    evaluate_model(model, X_val, y_val, threshold)
    explain_model(model, X_val, y_val)
    print("Training complete!")


if __name__ == "__main__":
    main()
