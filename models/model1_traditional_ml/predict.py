#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
=============================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model1_results.csv
"""
import pandas as pd
from pathlib import Path
from pipelines.data_pipeline import  get_data_and_process_target
# Paths
MODEL_PATH = Path("models/model1_traditional_ml/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"


import joblib

def load_model():
    # Load all the components you saved in train.py
    model = joblib.load(MODEL_PATH / "model.joblib")
    scaler = joblib.load(MODEL_PATH / "scaler.joblib")
    le = joblib.load(MODEL_PATH / "label_encoder.joblib")
    features = joblib.load(MODEL_PATH / "feature_columns.joblib")
    return model, scaler, le, features

def predict(model, scaler, feature_cols, test_df):
    # 1. Apply your custom pipeline steps
    from pipelines.data_pipeline import clean_data
    from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
    
    processed_df = clean_data(test_df)
    processed_df = accident_engineer_features(processed_df)
    
    # 2. Match the columns exactly to what the model expects
    processed_df = processed_df.reindex(columns=feature_cols, fill_value=0)
    
    # 3. Scale the data using the training scaler
    X_scaled = scaler.transform(processed_df)
    
    # 4. Get predictions and probabilities
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled).max(axis=1) # Highest probability as confidence
    
    return preds, probs


def main():
    # Load model
    model = load_model()

    # Load test data
    # TODO: Update this path to match your test data file
    # test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    # Generate predictions
    # predictions = predict(model, test_df)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "prediction": predictions,
    #     "probability": model.predict_proba(X_test)[:, 1],
    #     "confidence": confidence_scores,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
