#!/usr/bin/env python3
"""
Prediction Script for Model 2 (DNN)
"""

import sys
import joblib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

MODEL_DIR = CURRENT_DIR
# ---------------------------------------------------------------------
# Load bundle + model
# ---------------------------------------------------------------------
bundle_path = MODEL_DIR / "bundle.pkl"

if not bundle_path.exists():
    raise FileNotFoundError(f"Bundle not found at: {bundle_path}")

bundle = joblib.load(bundle_path)

model = tf.keras.models.load_model(bundle["keras_model_path"])
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
selected_features = bundle["selected_features"]

print("Model + bundle loaded")

# ---------------------------------------------------------------------
# Load pipeline functions (same logic as training)
# ---------------------------------------------------------------------
PIPELINE_MODULES = [
    "pipelines.data_pipeline",
    "pipelines.data_cleaning_accident_pipeline",
]

def load_pipeline_functions():
    funcs = {}

    for module_name in PIPELINE_MODULES:
        try:
            module = importlib.import_module(module_name)

            for name in ["clean_data", "accident_engineer_features"]:
                if hasattr(module, name) and name not in funcs:
                    funcs[name] = getattr(module, name)

        except:
            continue

    return funcs


FUNC = load_pipeline_functions()

clean_data = FUNC.get("clean_data")
accident_engineer_features = FUNC.get("accident_engineer_features")

# ---------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------
def preprocess(df):
    print("\n--- Preprocessing New Data ---")

    # Clean
    if clean_data:
        df = clean_data(df)

    # Feature engineering (safe)
    if accident_engineer_features:
        try:
            df = accident_engineer_features(df)
        except Exception as e:
            print(f" Feature engineering skipped: {e}")

    # Keep only numeric
    X = df.select_dtypes(include=[np.number])

    # Align columns to training
    missing_cols = [c for c in selected_features if c not in X.columns]

    for col in missing_cols:
        X[col] = 0  # fill missing columns

    # Ensure same order
    X = X[selected_features]

    print(f"Final input shape: {X.shape}")

    # Scale
    X_scaled = scaler.transform(X)

    return X_scaled


def predict(df):
    X = preprocess(df)

    proba = model.predict(X)

    preds = np.argmax(proba, axis=1)
    labels = label_encoder.inverse_transform(preds)

    return labels, proba


# ---------------------------------------------------------------------
# Run example
# --------------------------2-------------------------------------------
def main():
    # CHANGE THIS to your file
    input_file = CURRENT_DIR / "new_data.csv"
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    labels, proba = predict(df)

    # Save results
    df["prediction"] = labels

    output_path = CURRENT_DIR / "predictions.csv"
    df.to_csv(output_path, index=False)

    print(f"\n Predictions saved to: {output_path}")
    print(df[["prediction"]].head())


if __name__ == "__main__":
    main()