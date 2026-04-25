#!/usr/bin/env python3
"""
Model 2: Deep Neural Network — Stable Tabular Training Script
"""

import sys
import warnings
import joblib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SAVED_MODEL_DIR = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# SAFE TABULAR PIPELINE LOADER
# ---------------------------------------------------------------------

PIPELINE_MODULES = [
    "pipelines.data_pipeline",
    "pipelines.data_cleaning_accident_pipeline",
    "pipelines.Classification_pipelines",
    "pipelines.Regression_pipelines",
]

REQUIRED_FUNCTIONS = [
    "clean_data",
    "drop_low_variance_columns",
    "load_raw_data",
    "print_model_report",
]

OPTIONAL_FUNCTIONS = [
    "accident_engineer_features",
    "label_encode_target",
    "split_data",
]


def load_pipeline_functions():
    function_map = {}

    print("\nLoading TABULAR pipeline functions...\n")

    for module_name in PIPELINE_MODULES:
        try:
            module = importlib.import_module(module_name)
            print(f"Loaded: {module_name}")

            for func in REQUIRED_FUNCTIONS + OPTIONAL_FUNCTIONS:
                if hasattr(module, func) and func not in function_map:
                    function_map[func] = getattr(module, func)
                    print(f"   ↳ Found: {func}")

        except Exception as e:
            print(f"Skipped {module_name}: {e}")

    missing = [f for f in REQUIRED_FUNCTIONS if f not in function_map]
    if missing:
        raise ImportError(f"\n❌ Missing required functions: {missing}")

    return function_map


FUNC = load_pipeline_functions()

clean_data = FUNC["clean_data"]
drop_low_variance_columns = FUNC["drop_low_variance_columns"]
load_raw_data = FUNC["load_raw_data"]
print_model_report = FUNC["print_model_report"]

accident_engineer_features = FUNC.get("accident_engineer_features")
label_encode_target = FUNC.get("label_encode_target")
split_data = FUNC.get("split_data")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CLASS_WEIGHTS = {0: 2.0, 1: 1.0, 2: 1.5, 3: 2.0}
THRESHOLD_CLASS0 = 0.30
THRESHOLD_CLASS3 = 0.40


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_data():
    df = load_raw_data("city_traffic_accidents.csv")
    print(f"Loaded data: {df.shape}")
    return df


def preprocess_features(df):
    print("\n--- Preprocessing ---")

    # Clean
    df = clean_data(df)

    # Feature Engineering (SAFE)
    if accident_engineer_features:
        try:
            df = accident_engineer_features(df)
            print(" Feature engineering applied")
        except Exception as e:
            print(f" Feature engineering skipped: {e}")

    # Drop low variance
    df = drop_low_variance_columns(df)

    target_col = "Severity"

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found")

    df = df[df[target_col].notna()].copy()
    df = df.dropna(axis=1)

    drop_cols = [target_col, "Start_Time", "End_Time", "Weather_Timestamp"]
    existing_drop = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop)
    y = df[target_col]

    # REMOVE NON-NUMERIC DATA (FIXES YOUR ERROR)
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.select_dtypes(include=[np.number])

    print(f"Final feature shape: {X.shape}")

    # Encode target
    if label_encode_target:
        try:
            y_encoded, encoder = label_encode_target(y)
            print("Used pipeline encoder")
        except Exception:
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            print("Fallback encoder used")
    else:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

    # Split
    if split_data:
        try:
            X_train, X_val, y_train, y_val = split_data(X, y_encoded)
            print("Used pipeline split")
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            print("Fallback split used")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    artifacts = {
        "scaler": scaler,
        "label_encoder": encoder,
        "selected_features": X.columns.tolist(),
        "target_name": target_col,
    }

    return X_train, X_val, y_train, y_val, artifacts


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(input_dim):
    tf.keras.backend.clear_session()

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(4, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(model, X_train, y_train):
    print("\n--- Training ---")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=1024,
        class_weight=CLASS_WEIGHTS,
        callbacks=[early_stop],
        verbose=1,
    )


def evaluate_model(model, X_val, y_val, artifacts):
    print("\n--- Evaluation ---")

    proba = model.predict(X_val)

    preds = []
    for p in proba:
        if p[0] >= THRESHOLD_CLASS0:
            preds.append(0)
        elif p[3] >= THRESHOLD_CLASS3:
            preds.append(3)
        else:
            preds.append(int(np.argmax(p)))

    preds = np.array(preds)

    print_model_report(y_val, preds, "DNN")

    print(classification_report(
        y_val,
        preds,
        target_names=[str(c) for c in artifacts["label_encoder"].classes_],
        zero_division=0
    ))


def save_model(model, artifacts):
    print("\n--- Saving Model ---")

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = SAVED_MODEL_DIR / "model.keras"
    model.save(keras_path)

    bundle = {"keras_model_path": str(keras_path), **artifacts}
    joblib.dump(bundle, SAVED_MODEL_DIR / "bundle.pkl")

    print(f"Saved model to: {keras_path}")
    print(f"Saved bundle to: bundle.pkl")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = load_data()

    X_train, X_val, y_train, y_val, artifacts = preprocess_features(df)

    model = build_model(X_train.shape[1])

    train_model(model, X_train, y_train)

    evaluate_model(model, X_val, y_val, artifacts)

    save_model(model, artifacts)

    print("\nDONE — Model is trained and saved.")


if __name__ == "__main__":
    main()