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
    # 1. Load the model and its supporting artifacts
    model, scaler, le, feature_cols = load_model()

    # 2. Load the test data (Instructor will place this in test_data/)
    # For local testing, ensure city_traffic_accidents.csv is in test_data/
    test_file = TEST_DATA_DIR / "city_traffic_accidents.csv"
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        return

    test_df = pd.read_csv(test_file)

    # 3. Generate predictions
    # Note: Using the 'id' column from the original test data for the output
    preds, probs = predict(model, scaler, feature_cols, test_df)

    # 4. Format the output to match Model 1 template: id, prediction, probability, confidence
    results = pd.DataFrame({
        'id': test_df['id'],                 # Original record ID
        'prediction': le.inverse_transform(preds), # Convert 0/1/2 back to Severity labels
        'probability': probs,                # The raw probability
        'confidence': probs                  # In many cases, confidence = max probability
    })

    # 5. Save to the required location
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

    print(f"Predictions saved to {OUTPUT_FILE}")
