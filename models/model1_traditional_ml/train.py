#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
=========================================
Train an interpretable traditional ML model for accident severity classification.

Pipeline:
1. Load raw data
2. Clean + feature engineer
3. Split features/target
4. Encode target
5. Train/validation split
6. Scale features
7. Apply SMOTE on training set only
8. Train XGBoost
9. Evaluate
10. Explain with permutation importance
11. Save model bundle
"""

import sys
import warnings
import joblib
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")

# ---------------------------------------------------------------------
# Shared pipeline imports
# ---------------------------------------------------------------------
from pipelines.data_pipeline import (
    load_raw_data,
    clean_data,
    accident_engineer_features,
    drop_low_variance_columns,
    split_data,
    scale_features,
    label_encode_target,
    print_model_report,
    plot_feature_importance,
)

warnings.filterwarnings("ignore")


def load_data():
    """
    Load raw accident data from data/raw/.
    """
    df = load_raw_data("city_traffic_accidents.csv")
    print(f"Loaded raw data: {df.shape}")
    return df


def preprocess_features(df):
    """
    Full preprocessing pipeline:
    - clean raw data
    - feature engineering
    - drop low-variance columns
    - separate X and y
    - encode target labels
    - train/validation split
    - scale numeric features
    - SMOTE on training data only

    Returns:
        X_train_res, X_val_scaled, y_train_res, y_val, artifacts
    """
    print("\n--- Preprocessing ---")

    # 1. Clean + engineer
    df = clean_data(df)
    df = accident_engineer_features(df)
    df = drop_low_variance_columns(df)

    target_col = "Severity"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # 2. Keep only rows with target
    df = df[df[target_col].notna()].copy()

    # 3. Drop columns with any remaining nulls
    #    This is simple and safe for now; can be replaced with targeted imputation later
    df = df.dropna(axis=1)

    print(f"Post-processing dataframe shape: {df.shape}")

    # 4. Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 5. Encode target labels
    y_encoded, label_encoder = label_encode_target(y)

    # 6. Split train / validation
    X_train, X_val, y_train, y_val = split_data(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # 7. Scale features
    X_train_scaled, X_val_scaled, scaler, selected_features = scale_features(
        X_train, X_val
    )

    # 8. SMOTE on training only
    print("\n--- SMOTE Component ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"Before SMOTE: X_train={X_train_scaled.shape}, y_train={len(y_train)}")
    print(f"After SMOTE : X_train={X_train_res.shape}, y_train={len(y_train_res)}")

    artifacts = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "selected_features": selected_features,
        "target_name": target_col,
    }

    return X_train_res, X_val_scaled, y_train_res, y_val, artifacts


def train_model(X_train, y_train):
    """
    Train XGBoost classifier.
    """
    print("\n--- Model Training ---")

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("Model training complete.")

    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance on validation data.
    """
    print("\n--- Evaluation ---")

    y_pred = model.predict(X_val)

    print_model_report(y_val, y_pred, "XGBoost Classifier")

    acc = accuracy_score(y_val, y_pred)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"Accuracy    : {acc:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    metrics = {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
    }

    return metrics


def explain_model(model, X_val, y_val):
    """
    Generate feature importance analysis using permutation importance.
    """
    print("\n--- Model Explainability ---")

    feat_imp = plot_feature_importance(
        trained_model=model,
        X_val=X_val,
        y_val=y_val,
        model_name="XGBoost Classifier",
        top_n=30,
    )

    return feat_imp


def save_model(model, artifacts):
    """
    Save trained model bundle, scaler, label encoder, and selected features.
    """
    print("\n--- Saving Model Bundle ---")

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": artifacts["scaler"],
        "label_encoder": artifacts["label_encoder"],
        "selected_features": artifacts["selected_features"],
        "target_name": artifacts["target_name"],
    }

    output_path = SAVED_MODEL_DIR / "EW_xgb_model.pkl"
    joblib.dump(bundle, output_path)

    print(f"Saved model bundle to: {output_path}")


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess
    X_train, X_val, y_train, y_val, artifacts = preprocess_features(df)

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # 5. Explain
    explain_model(model, X_val, y_val)

    # 6. Save
    save_model(model, artifacts)

    print("\nTraining complete!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
