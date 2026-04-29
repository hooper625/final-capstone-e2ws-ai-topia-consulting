#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
===========================================
Train a classical ML model (XGBoost, Random Forest, etc.) on your scenario's
tabular data.

IMPORTANT: This model must be interpretable. Include SHAP or feature importance
analysis so stakeholders can understand WHY the model makes its predictions.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")
#Import
from pipelines.data_pipeline import load_raw_data, clean_data, save_processed_data, drop_low_variance_columns, get_data_and_process_target, split_data, scale_features
from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
from pipelines.Classification_pipelines import plot_feature_importance

def load_data():
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    #Load the City Traffic Accident Database
    df = load_raw_data("city_traffic_accidents.csv")
    return df


def preprocess_features(df):
    """Select and prepare features for training."""
    TARGET = 'Severity'
    processed_path = PROCESSED_DATA / "city_traffic_processed.csv"

    if not processed_path.exists():
        # Full pipeline — only needed the first time or after raw data changes
        UI_FEATURES = [
            'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'Distance(mi)', 'n_road_features', 'has_traffic_control',
            'is_freezing', 'low_visibility_severity', 'has_precipitation',
            'weather_cluster_clear', 'weather_cluster_cloudy',
            'weather_cluster_low_visibility', 'weather_cluster_rain',
            'weather_cluster_snow_ice', 'DangerousScore',
        ]
        df = clean_data(df)
        df = accident_engineer_features(df)
        df = drop_low_variance_columns(df)
        df = df.dropna(axis=1)
        available = [c for c in UI_FEATURES if c in df.columns]
        df = df[[TARGET] + available]
        save_processed_data(df, "city_traffic_processed.csv")
    else:
        print("Processed data found — skipping feature engineering.")

    df, target_stats = get_data_and_process_target("city_traffic_processed.csv", target_column=TARGET)
    if target_stats:
        print(f"\nReady to process models for {TARGET}...")

    X = df.drop(columns=[TARGET]).copy()

    # Binary target: High Risk (Severity 3+4) = 1, Standard Risk (Severity 1+2) = 0
    y = (df[TARGET] >= 3).astype(int)
    print(f"  Standard Risk (0): {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"  High Risk     (1): {(y==1).sum():,} ({(y==1).mean():.1%})")

    label_map = {0: "Standard Risk", 1: "High Risk"}

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler, features = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_map, features


def train_model(X_train, y_train):
    # Subsample to 80k rows — sufficient for 16 features and much faster to train
    if len(X_train) > 80_000:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=80_000, random_state=42, stratify=y_train
        )
        print(f"Subsampled to {len(y_train):,} rows for training.")

    sample_weights = compute_sample_weight('balanced', y_train)

    mdl = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        n_jobs=-1,
        random_state=42,
        objective='binary:logistic',
    )
    mdl.fit(X_train, y_train, sample_weight=sample_weights)
    return mdl


def evaluate_model(model, X_val, y_val):
    # Evaluate only — never re-fit, which would erase the sample weights
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_val)
    print(f"Test Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"Test F1 (wtd) : {f1_score(y_val, y_pred, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred))
    return {"accuracy": accuracy_score(y_val, y_pred),
            "f1_weighted": f1_score(y_val, y_pred, average='weighted')}


def explain_model(model, X_val, y_val):
    print("\n--- Feature Importance Analysis ---")
    plot_feature_importance(model, X_val, y_val, "XGBoost")


def save_model(model, scaler, le, feature_cols):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(scaler,       SAVED_MODEL_DIR / "scaler.joblib")
    joblib.dump(le,           SAVED_MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_columns.joblib")


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess features
    X_train, X_val, y_train, y_val, scaler, le, feature_cols = preprocess_features(df)

    # 3. Train model
    model = train_model(X_train, y_train)

    # 4. Save BEFORE evaluate so the weighted model is never overwritten
    save_model(model, scaler, le, feature_cols)

    # 5. Evaluate (predict-only, no re-fit)
    evaluate_model(model, X_val, y_val)

    # 6. Explain — REQUIRED
    explain_model(model, X_val, y_val)

    print("Training complete!")


if __name__ == "__main__":
    main()
