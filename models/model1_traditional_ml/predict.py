#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
===========================================
Loads the trained XGBoost model bundle and generates predictions on test data.

Usage examples:
    python -m models.model1_traditional_ml.predict
    python -m models.model1_traditional_ml.predict --input test_data/test_data_file.csv
    python -m models.model1_traditional_ml.predict --self-test

Output:
    test_data/model1_results.csv
"""

import sys
import argparse
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------------------------------------------------------------------
# Shared pipeline imports
# ---------------------------------------------------------------------
from pipelines.data_pipeline import (
    clean_data,
    accident_engineer_features,
    drop_low_variance_columns,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
MODEL_PATH = Path("models/model1_traditional_ml/saved_model/EW_xgb_model.pkl")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"


def load_model():
    """
    Load trained model bundle from .pkl file.

    Expected bundle keys:
        - model
        - scaler
        - label_encoder
        - selected_features
        - target_name
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Make sure EW_xgb_model.pkl exists."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = ["model", "scaler", "label_encoder", "selected_features", "target_name"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise KeyError(f"Model bundle is missing keys: {missing}")

    print(f"Loaded model bundle from: {MODEL_PATH}")
    return bundle


def preprocess_for_prediction(df, selected_features, target_name="Severity"):
    """
    Apply the same feature engineering used in training, then align columns
    to the exact selected feature list used by the model.

    Important:
    - We do NOT fit anything here.
    - We DO add missing columns as 0.
    - We DO drop extra columns not seen during training.
    """
    df = df.copy()

    # Keep original id if present
    id_series = df["id"].copy() if "id" in df.columns else pd.Series(range(len(df)), name="id")

    # Remove target column if present in scoring data
    if target_name in df.columns:
        df = df.drop(columns=[target_name], errors="ignore")

    # Standard preprocessing
    df = clean_data(df)
    df = accident_engineer_features(df)
    df = drop_low_variance_columns(df)

    # Drop any columns with all-null values
    df = df.dropna(axis=1, how="all")

    # Align to training features
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only training features, in exact order
    X = df.reindex(columns=selected_features, fill_value=0)

    # Final missing-value handling
    X = X.fillna(0)

    return id_series, X


def predict(bundle, test_df):
    """
    Generate predictions and probabilities.

    Returns DataFrame with:
        id, prediction, probability, confidence
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    selected_features = bundle["selected_features"]
    target_name = bundle["target_name"]

    ids, X = preprocess_for_prediction(
        test_df,
        selected_features=selected_features,
        target_name=target_name
    )

    # Scale using training-fitted scaler
    X_scaled = scaler.transform(X)

    # Predict encoded classes
    pred_encoded = model.predict(X_scaled)

    # Convert back to original labels
    pred_labels = label_encoder.inverse_transform(pred_encoded.astype(int))

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        max_probs = probs.max(axis=1)
    else:
        probs = None
        max_probs = np.full(shape=len(pred_labels), fill_value=np.nan)

    # Simple confidence band
    confidence = pd.cut(
        max_probs,
        bins=[-0.01, 0.50, 0.75, 0.90, 1.00],
        labels=["low", "medium", "high", "very_high"]
    ).astype(str)

    results = pd.DataFrame({
        "id": ids.values,
        "prediction": pred_labels,
        "probability": max_probs,
        "confidence": confidence
    })

    return results


def make_sample_test_cases():
    """
    Generate a couple of simple test cases for quick smoke testing.

    These are illustrative only. Replace with real test data for production.
    """
    sample_df = pd.DataFrame([
        {
            "id": 1,
            "Start_Time": "2024-01-15 08:15:00",
            "End_Time": "2024-01-15 09:05:00",
            "Weather_Timestamp": "2024-01-15 08:00:00",
            "State": "WA",
            "Zipcode": "98101",
            "City": "Seattle",
            "County": "King",
            "Start_Lat": 47.6097,
            "Start_Lng": -122.3331,
            "End_Lat": 47.6100,
            "End_Lng": -122.3340,
            "Temperature(F)": 34.0,
            "Humidity(%)": 88.0,
            "Pressure(in)": 29.8,
            "Visibility(mi)": 1.5,
            "Wind_Speed(mph)": 18.0,
            "Precipitation(in)": 0.2,
            "Weather_Condition": "Light Rain",
            "Sunrise_Sunset": "Night",
            "Astronomical_Twilight": "Night",
            "Wind_Direction": "SW",
            "Amenity": False,
            "Bump": False,
            "Crossing": True,
            "Give_Way": False,
            "Junction": True,
            "No_Exit": False,
            "Railway": False,
            "Roundabout": False,
            "Station": False,
            "Stop": True,
            "Traffic_Calming": False,
            "Traffic_Signal": True,
            "Turning_Loop": False,
            "Description": "Accident on freeway near exit ramp with wet road conditions.",
            "Country": "US",
            "ID": "A1",
            "Source": "Test"
        },
        {
            "id": 2,
            "Start_Time": "2024-07-10 14:20:00",
            "End_Time": "2024-07-10 14:40:00",
            "Weather_Timestamp": "2024-07-10 14:00:00",
            "State": "AZ",
            "Zipcode": "85007",
            "City": "Phoenix",
            "County": "Maricopa",
            "Start_Lat": 33.4484,
            "Start_Lng": -112.0740,
            "End_Lat": 33.4488,
            "End_Lng": -112.0730,
            "Temperature(F)": 104.0,
            "Humidity(%)": 22.0,
            "Pressure(in)": 29.7,
            "Visibility(mi)": 10.0,
            "Wind_Speed(mph)": 6.0,
            "Precipitation(in)": 0.0,
            "Weather_Condition": "Clear",
            "Sunrise_Sunset": "Day",
            "Astronomical_Twilight": "Day",
            "Wind_Direction": "N",
            "Amenity": False,
            "Bump": False,
            "Crossing": False,
            "Give_Way": False,
            "Junction": False,
            "No_Exit": False,
            "Railway": False,
            "Roundabout": False,
            "Station": False,
            "Stop": False,
            "Traffic_Calming": False,
            "Traffic_Signal": False,
            "Turning_Loop": False,
            "Description": "Minor collision on a dry straight road in clear weather.",
            "Country": "US",
            "ID": "A2",
            "Source": "Test"
        }
    ])

    return sample_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Path to output CSV file")
    parser.add_argument("--self-test", action="store_true", help="Run prediction on generated sample test cases")
    args = parser.parse_args()

    # Load model bundle
    bundle = load_model()

    # Load test data
    if args.self_test:
        test_df = make_sample_test_cases()
        print(f"Running self-test with {len(test_df)} sample rows.")
    else:
        if args.input is None:
            raise ValueError(
                "No input file provided. Use --input <path_to_csv> or --self-test."
            )
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        test_df = pd.read_csv(input_path)
        print(f"Loaded test data: {test_df.shape} from {input_path}")

    # Predict
    results = predict(bundle, test_df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nPrediction preview:")
    print(results.head())

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
