#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
===========================================
Train a classical ML model (XGBoost, Random Forest, etc.) on your scenario's
tabular data.

IMPORTANT: This model must be interpretable. Include SHAP or feature importance
analysis so stakeholders can understand WHY the model makes its predictions.
"""
from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")
#Import
from pipelines.data_pipeline import load_raw_data, clean_data, accident_engineer_features, save_processed_data, drop_low_variance_columns
from pipelines.data_pipeline import generate_hourly_heatmap, generate_accident_map # functions to create maps

def load_data():
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    #Load the City Traffic Accident Database
    df = load_raw_data("city_traffic_accidents.csv")
    return df
    raise NotImplementedError


def preprocess_features(df):
    """Select and prepare features for training.

    Consider:
    - Feature selection (drop leaky or irrelevant columns)
    - Encoding categorical variables
    - Scaling numerical features
    - Handling missing values
    """
    df = clean_data(df)                       #Clean the data (handle missing values, convert data types, etc.)
    df = accident_engineer_features(df)       #Engineer features specific to traffic accidents (e.g., severity, weather conditions, etc.)
    #Generate the heatmap and accident map for City Traffic Accident
    generate_hourly_heatmap(df)                            #Generate a heatmap to visualize the density of accidents over time and location
    generate_accident_map(df)                              #Generate a map to visualize the locations of
    df = drop_low_variance_columns(df)
    df = df.dropna(axis=1) 

    return df

    raise NotImplementedError


def train_model(X_train, y_train):
    """Train your traditional ML model.

    Recommended algorithms:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier

    IMPORTANT: Handle class imbalance!
        model = RandomForestClassifier(class_weight='balanced')
    """
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance on validation data.

    Must include:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix
    - Weighted F1 score (primary metric for imbalanced data)
    - AUC-ROC (for binary classification scenarios)
    """
    # TODO: Print evaluation metrics
    raise NotImplementedError


def explain_model(model, X_val):
    """Generate SHAP or feature importance analysis.

    This is REQUIRED — your model must be interpretable.

    Option 1 — SHAP (recommended):
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.summary_plot(shap_values, X_val)

    Option 2 — Built-in feature importance:
        importances = model.feature_importances_
        # Plot top 15 features
    """
    # TODO: Generate explainability analysis
    raise NotImplementedError


def save_model(model):
    """Save the trained model to saved_model/.

    Example:
        import joblib
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    """
    save_processed_data(model, "model.joblib")
    raise NotImplementedError


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess features
    X_train, X_val, y_train, y_val = preprocess_features(df)

    # 3. Train model
    model = train_model(X_train, y_train)

    # 4. Evaluate
    evaluate_model(model, X_val, y_val)

    # 5. Explain — REQUIRED
    explain_model(model, X_val)

    # 6. Save
    save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()