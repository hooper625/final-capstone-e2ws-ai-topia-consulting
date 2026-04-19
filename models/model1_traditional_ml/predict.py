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


def load_model():
    """Load your trained model from saved_model/.

    Typical approaches:
        import joblib
        model = joblib.load(MODEL_PATH / "model.joblib")

    Works with XGBoost, Random Forest, Logistic Regression, etc.
    """
    # Initialize your app variables
    TARGET = 'Severity'

    # Use the component to load data
    df, target_stats = get_data_and_process_target("city_traffic_processed.csv", target_column=TARGET)
    raise NotImplementedError("Load your trained model here")


def predict(model, test_data):
    """Generate predictions on test data.

    Should return a DataFrame with columns: id, prediction, probability, confidence
    """
    # TODO: Run your model on the test data
    raise NotImplementedError("Generate predictions here")


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
