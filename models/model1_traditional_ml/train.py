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
from imblearn.over_sampling import SMOTE

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")
#Import
from pipelines.data_pipeline import load_raw_data, clean_data, save_processed_data, drop_low_variance_columns, get_data_and_process_target, label_encode_target, split_data, scale_features
from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
from pipelines.Classification_pipelines import evaluate_classification_model, run_hist_gradient_boosting, run_random_forest, run_decision_tree,run_gradient_boosting, run_knn, run_svm_linear, run_voting_classifier, plot_feature_importance, run_xgb_classifier_feature

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
    """Select and prepare features for training.

    Consider:
    - Feature selection (drop leaky or irrelevant columns)
    - Encoding categorical variables
    - Scaling numerical features
    - Handling missing values
    """
    TARGET = 'Severity'

    df = clean_data(df)                                     #Clean the data (handle missing values, convert data types, etc.)
    df = accident_engineer_features(df)                     #Engineer features specific to traffic accidents (e.g., severity, weather conditions, etc.)

    df = drop_low_variance_columns(df)
    df = df.dropna(axis=1) 
    save_processed_data(df, "city_traffic_processed.csv")

    # Use the component to load data
    df, target_stats = get_data_and_process_target("city_traffic_processed.csv", target_column=TARGET)
    if target_stats:
        print(f"\nReady to process models for {TARGET}...")
    
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET]

    # 2. Encode and Split
    y_encoded, le = label_encode_target(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded, test_size=0.2)

    # 3. Scale (Crucial: Keep the Scaled versions!)
    X_train_scaled, X_test_scaled, scaler, features = scale_features(X_train, X_test)

    # 4. SMOTE on Scaled training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Return scaled training AND scaled test/validation data
    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, le, features


def train_model(X_train, X_test, y_train, y_test):
    _, mdl = run_xgb_classifier_feature(
        X_train, X_test, y_train, y_test,
        max_depth=6,
        n_estimators=200
    )
    return mdl


def evaluate_model(model, X_train, X_val, y_train, y_val):
    print("\n--- Model Evaluation ---")
    metrics, _, _ = evaluate_classification_model(model, X_train, X_val, y_train, y_val, "XGBoost")
    return metrics


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
    model = train_model(X_train, X_val, y_train, y_val)

    # 4. Evaluate
    evaluate_model(model, X_train, X_val, y_train, y_val)

    # 5. Explain — REQUIRED
    explain_model(model, X_val, y_val)

    # 6. Save
    save_model(model, scaler, le, feature_cols)

    print("Training complete!")


if __name__ == "__main__":
    main()
