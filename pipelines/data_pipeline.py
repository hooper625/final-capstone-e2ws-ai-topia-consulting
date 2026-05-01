"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization
import matplotlib.pyplot as plt
from seaborn.objects import Plot
from sklearn.inspection import permutation_importance

# Sklearn - evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, classification_report, confusion_matrix,  RocCurveDisplay, PrecisionRecallDisplay
)
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# =============================================================================
# HINT 1: Loading the Accident Data
# =============================================================================
def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_encounters_2023.csv")

    Returns:
        pandas DataFrame

    Example:
        df = load_raw_data("patient_encounters_2023.csv")
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    
    return pd.read_csv(filepath)

# ============================================================================================================================
# Common Data Cleaning & Feature Engineering
# ============================================================================================================================
def clean_data(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    #drop Duplicates
    df = df.drop_duplicates()

    # cleaning - change all text to lower case for consistency
    df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

    return df

# =============================================================================
# HINT 2: Temporal Feature Engineering
# =============================================================================
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time patterns are among the strongest predictors of accident severity.

    Features to extract:
    - Hour of day (rush hour vs. off-peak)
    - Day of week (weekday vs. weekend)
    - Month (seasonal patterns — winter ice, summer heat)
    - Duration of traffic impact
    - Is it dark? (Sunrise_Sunset column helps, but you can derive from time too)
    """
    if 'Start_Time' in df.columns:
        df['hour'] = df['Start_Time'].dt.hour
        df['day_of_week'] = df['Start_Time'].dt.dayofweek
        df['month'] = df['Start_Time'].dt.month
    
    if 'day_of_week' in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Rush hour flags
    if 'hour' in df.columns:
        df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
        df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)

    if 'is_morning_rush' in df.columns and 'is_evening_rush' in df.columns:
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # Duration of traffic impact (in minutes)
    if 'End_Time' in df.columns:
        df['duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        # Cap extreme values
        df['duration_min'] = df['duration_min'].clip(0, 1440)  # Max 24 hours

    # Handle Created Date
    if 'created_date' in df.columns:
        # Convert to datetime first to avoid the AttributeError
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        
        # Use specific names so they don't get overwritten
        df['created_hour'] = df['created_date'].dt.hour
        df['created_day_of_week'] = df['created_date'].dt.dayofweek
        df['created_month'] = df['created_date'].dt.month

    # Handle Closed Date
    if 'closed_date' in df.columns:
        # Convert to datetime first
        df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
        
        # Use 'closed_' prefix
        df['closed_hour'] = df['closed_date'].dt.hour
        df['closed_day_of_week'] = df['closed_date'].dt.dayofweek
        df['closed_month'] = df['closed_date'].dt.month
        
    return df

# =============================================================================
# Split data into train and validation sets
# =============================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    Verifies the balance of the split automatically.
    """
    # 1. Perform the split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # 2. Verify split size (Your requested check)
    print("--- Data Split Component ---")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 3. Verify stratification/class distribution
    print(f"\nTraining class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        percentage = (c / len(y_train)) * 100
        print(f" Class {u}: {c} samples ({percentage:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# =============================================================================
# Save processed data 
# =============================================================================
def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename

    # Drop the file if it already exists
    if output_path.exists():
        output_path.unlink()
        print(f"Existing file {filename} dropped.")

    # Save the new version
    df.to_csv(output_path, index=False)
    print(f"Saved fresh processed data to {output_path}")

# =============================================================================
# Load previously processed data 
# =============================================================================
def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)

# =============================================================================
# Changes true and false to 1 and 0
# =============================================================================
def convert_bools_to_ints(df):
    # 1. Find all columns that are of type 'bool'
    bool_cols = df.select_dtypes(include=['bool']).columns
    
    # 2. Convert only those columns to integer (True -> 1, False -> 0)
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df

# ==================================================================================================
# Drop columns where a single value occupies more than 99% of the rows
# ==================================================================================================
def drop_low_variance_columns(df, threshold=0.99):
    """
    This function removes features that are functionally constant 
    (i.e., where a single value occupies more than the threshold percentage of the total rows),
    because a column where almost every row is the same provides no "contrast" for the model to learn from.
    """
    # Identify columns to drop
    cols_to_drop = []
    
    for col in df.columns:
        # Get the proportion of the most frequent value
        top_value_ratio = df[col].value_counts(normalize=True).iloc[0]
        
        if top_value_ratio > threshold:
            cols_to_drop.append(col)
            
    print(f"Dropped {len(cols_to_drop)} columns with > {threshold*100}% dominance.")
    print(f"Columns dropped: {cols_to_drop}")
    
    return df.drop(columns=cols_to_drop)

# ==================================================================================================
# gets the process data and prints out the range and std of the target variable for interpretation
# ==================================================================================================
def get_data_and_process_target(file_path, target_column):
    """
    Loads and inspects the processed dataset.
    """
    try:
        # Load the CSV
        load_processed_data(file_path)
        df = load_processed_data(file_path)
        
        # Basic inspection
        print(f"--- Data Successfully Loaded ---")
        print(f"Data shape: {df.shape}")
        
        # Calculate target stats for interpretation
        if target_column in df.columns:
            target_range = df[target_column].max() - df[target_column].min()
            target_std = df[target_column].std()
            
            print(f"Target Column: '{target_column}'")
            print(f"Target range: {target_range:,.2f}")
            print(f"Target std: {target_std:,.2f}")
            
            # Return the dataframe and the stats as a dictionary
            stats = {
                'range': target_range,
                'std': target_std
            }
            return df, stats
        else:
            print(f"Error: Target column '{target_column}' not found in dataframe.")
            return df, None
            
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

# ==================================================================================================
# Feature Scaling   
# ==================================================================================================
def scale_features(X_train, X_test):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Returns DataFrames to preserve column names and indices.
    """
    # 1. Initialize the Scaler
    scaler = StandardScaler()
    
    # 2. Fit on TRAIN and Transform BOTH
    # (We only 'fit' on train to prevent data leakage from the test set)
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # 3. Convert back to DataFrames to keep columns and index
    X_train_scaled = pd.DataFrame(
        X_train_scaled_array, 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array, 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # 4. Track metadata for the app
    SELECTED_FEATURES = X_train.columns.tolist()
    
    print("--- Scaling Component ---")
    print(f"Features scaled successfully!")
    print(f"Scaler fitted on {len(SELECTED_FEATURES)} features.")
    
    return X_train_scaled, X_test_scaled, scaler, SELECTED_FEATURES

# ==================================================================================================
# Label Encoding for Classification Targets
# ==================================================================================================
def label_encode_target(y):
    """
    Standardizes target labels to start at 0.
    Prints the mapping for verification.
    """
    # 1. Initialize the Encoder
    label_encoder = LabelEncoder()
    
    # 2. Fit and Transform the target
    y_encoded = label_encoder.fit_transform(y)
    
    # 3. Verify and Print encoding (Your requested check)
    print("--- Label Encoding Component ---")
    print("Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # Return the encoded array and the encoder object for inverse mapping later
    return y_encoded, label_encoder

# =============================================================================
# This helper function prints detailed diagnostic reports for the model
# =============================================================================
def print_model_report(y_test, y_test_pred, model_name):
    """Prints the official Scikit-Learn Classification 
    Example Output:
                  precision    recall  f1-score   support

               1       0.73      0.15      0.25       870
               2       0.85      0.96      0.91     79534
               3       0.70      0.39      0.50     16808
               4       0.62      0.11      0.18      2646

        accuracy                           0.84     99858
       macro avg       0.72      0.40      0.46     99858
    weighted avg       0.82      0.84      0.81     99858
    """
    print(f"\n" + "="*60)
    print(f" MODEL: {model_name}")
    print("="*60)
    
    # This produces the exact precision, recall, f1-score, support table
    # We don't hardcode target_names so it stays dynamic for different datasets
    report = classification_report(y_test, y_test_pred)
    print(report)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("="*60 + "\n")

# =============================================================================
# This helper function extracts, prints, and plots feature importances
# =============================================================================
def plot_feature_importance(trained_model, X_val, y_val, model_name, top_n=30):
    """
    Calculates Permutation Importance for any model type.
    Works for HistGradientBoosting, SVM, KNN, and Tree-based models.
    """
    print(f"\nCalculating feature importance for {model_name}...")
    
    # Calculate Permutation Importance
    # We use the validation/test set here to see which features actually matter for prediction
    result = permutation_importance(
       trained_model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1
    )

    # Map to column names
    feat_imp = pd.DataFrame({
        "feature": X_val.columns if isinstance(X_val, pd.DataFrame) else [f"Feature {i}" for i in range(X_val.shape[1])],
        "importance": result.importances_mean
    })

    # Sort and filter top N
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(top_n)

    # Print results
    print(f"Top {top_n} features for {model_name}:")
    for i, row in feat_imp.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color='teal')
    plt.xlabel("Decrease in Accuracy (Permutation Importance)")
    plt.ylabel("Feature")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return feat_imp

# =============================================================================
# This helper function plots how confident the model is in its predictions
# =============================================================================
def plot_prediction_probabilities(trained_model, X_test, model_name):
    """Plots a histogram of the max predicted probabilities for each sample."""
    # Check if model supports probability estimates
    if not hasattr(trained_model, "predict_proba"):
        print(f"{model_name} does not support predict_proba.")
        return

    # Get the probabilities for each class
    probs = trained_model.predict_proba(X_test)
    
    # Take the highest probability for each prediction (the confidence level)
    max_probs = np.max(probs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=30, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random Guessing (50%)')
    plt.xlabel('Predicted Probability (Confidence)')
    plt.ylabel('Number of Samples')
    plt.title(f'Prediction Confidence Distribution - {model_name}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


# =============================================================================
# Accident Feature Engineering — Prediction Pipeline
# Computes the 16 features Model 1 (XGBoost) expects.
# No KMeans, no uszipcode lookups, no dynamic OHE — safe on any dataset size.
# =============================================================================
def accident_predict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lean feature engineering for Model 1 prediction.
    Imports the weather/road helpers from data_cleaning_accident_pipeline
    at call time to avoid circular imports.
    """
    from pipelines.data_cleaning_accident_pipeline import (
        process_weather_features,
        dangerous_conditions_score,
        engineer_road_features,
    )

    df = df.copy()
    df = df.drop(columns=['Country', 'ID', 'Source'], errors='ignore')

    # ── Datetime parsing ──────────────────────────────────────────────────────
    for col in ['Start_Time', 'End_Time', 'Weather_Timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'Start_Time' in df.columns and 'Weather_Timestamp' in df.columns:
        df['Start_Time'] = df['Start_Time'].fillna(df['Weather_Timestamp'])

    # ── Temporal features ─────────────────────────────────────────────────────
    if 'Start_Time' in df.columns:
        df['hour']        = df['Start_Time'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['Start_Time'].dt.dayofweek.fillna(0).astype(int)
    else:
        df['hour']        = 12
        df['day_of_week'] = 0

    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)
    df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour']    = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # ── Accident duration (strong severity signal) ────────────────────────────
    if 'Start_Time' in df.columns and 'End_Time' in df.columns:
        df['duration_min'] = (
            (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        ).clip(0, 1440).fillna(0)
    else:
        df['duration_min'] = 0

    # ── Fill missing weather with global medians ──────────────────────────────
    for col in ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if 'Wind_Chill(F)' in df.columns and 'Temperature(F)' in df.columns:
        df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(df['Temperature(F)'])

    for col in ['Weather_Condition', 'Wind_Direction']:
        if col in df.columns:
            mode = df[col].dropna().mode()
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else 'Clear')

    for col in ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']:
        if col in df.columns:
            df[col] = df[col].fillna('Day')

    # ── Convert road feature columns from bool/string to int ─────────────────
    road_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    ]
    bool_map = {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
    for col in road_cols:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(0).astype(int)

    # ── Weather features (is_freezing, low_visibility_severity, weather clusters) ──
    df = process_weather_features(df)

    # ── Dangerous conditions composite score ──────────────────────────────────
    df = dangerous_conditions_score(df)

    # ── Road aggregate features (n_road_features, has_traffic_control) ────────
    df = engineer_road_features(df)

    return df
