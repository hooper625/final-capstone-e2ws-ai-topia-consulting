#!/usr/bin/env python3
"""
Model 2: Traffic Accident Severity — Deep Neural Network
=========================================================
4-class severity classifier (Severity 1–4) trained on the shared processed CSV.

Improvements:
  - Lighter fixed class weights (not aggressive sqrt-balanced)
  - Residual architecture with skip connections
  - Raw weather numerics + granular time features for richer input
  - Per-class threshold grid search on validation split
  - Optimal thresholds saved to thresholds.joblib for use in predict.py

To run from project root:
    python -u models/model2_deep_learning/train.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT    = Path.cwd()
RAW_DATA_PATH   = PROJECT_ROOT / "data" / "raw" / "city_traffic_accidents.csv"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

# Features we want to use — processed + raw weather numerics + time granulars
TARGET_FEATURES = [
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
    'hour', 'day_of_week', 'month',
    'Distance(mi)', 'n_road_features', 'has_traffic_control',
    'is_freezing', 'low_visibility_severity', 'has_precipitation',
    'weather_cluster_clear', 'weather_cluster_cloudy',
    'weather_cluster_low_visibility', 'weather_cluster_rain',
    'weather_cluster_snow_ice', 'DangerousScore',
    'Temperature(F)', 'Visibility(mi)', 'Precipitation(in)', 'Wind_Speed(mph)',
    'duration_min',
]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model2_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_and_engineer(logger: logging.Logger) -> pd.DataFrame:
    from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
    from pipelines.data_pipeline import clean_data, drop_low_variance_columns

    logger.info("Loading raw data from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    # Preserve raw weather numerics before pipeline drops/aggregates them
    weather_cols = ['Temperature(F)', 'Visibility(mi)', 'Precipitation(in)', 'Wind_Speed(mph)']
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Temporal features before cleaning removes them
    for dt_col in ['Start_Time', 'End_Time']:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

    if 'Start_Time' in df.columns:
        df['hour']        = df['Start_Time'].dt.hour.fillna(0).astype(int)
        df['day_of_week'] = df['Start_Time'].dt.dayofweek.fillna(0).astype(int)
        df['month']       = df['Start_Time'].dt.month.fillna(1).astype(int)

    if 'Start_Time' in df.columns and 'End_Time' in df.columns:
        df['duration_min'] = (
            (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        ).clip(0, 1440).fillna(0)
    else:
        df['duration_min'] = 0

    # Run full feature engineering pipeline
    logger.info("Running feature engineering pipeline …")
    df_clean = clean_data(df.copy())
    df_eng   = accident_engineer_features(df_clean)
    df_eng   = drop_low_variance_columns(df_eng)

    # Merge back weather numerics + time granulars that pipeline may have dropped
    extra_cols = [c for c in weather_cols + ['hour', 'day_of_week', 'month', 'duration_min']
                  if c not in df_eng.columns]
    for col in extra_cols:
        if col in df.columns:
            df_eng[col] = df[col].values[:len(df_eng)]

    logger.info("Engineered shape: %s", df_eng.shape)
    return df_eng


def build_model(input_dim: int):
    import tensorflow as tf
    from tensorflow.keras.regularizers import l2

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    inp = tf.keras.Input(shape=(input_dim,))

    # Block 1
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 2 — residual skip
    skip = tf.keras.layers.Dense(128)(x)
    x    = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x    = tf.keras.layers.BatchNormalization()(x)
    x    = tf.keras.layers.Add()([x, skip])
    x    = tf.keras.layers.Activation("relu")(x)
    x    = tf.keras.layers.Dropout(0.3)(x)

    # Block 3 — residual skip
    skip = tf.keras.layers.Dense(64)(x)
    x    = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x    = tf.keras.layers.BatchNormalization()(x)
    x    = tf.keras.layers.Add()([x, skip])
    x    = tf.keras.layers.Activation("relu")(x)
    x    = tf.keras.layers.Dropout(0.2)(x)

    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def apply_thresholds(proba: np.ndarray, t0: float = 0.30, t3: float = 0.20) -> np.ndarray:
    """Priority decode: lower bar for rare classes 0 (Sev 1) and 3 (Sev 4)."""
    preds = []
    for row in proba:
        if row[0] >= t0:
            preds.append(0)
        elif row[3] >= t3:
            preds.append(3)
        else:
            preds.append(int(np.argmax(row)))
    return np.array(preds)


def optimise_thresholds(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Grid search t0 and t3 on the validation set to maximise weighted F1."""
    best = {"t0": 0.30, "t3": 0.20, "f1": 0.0}
    for t0 in np.arange(0.05, 0.65, 0.05):
        for t3 in np.arange(0.05, 0.55, 0.05):
            preds = apply_thresholds(proba, t0, t3)
            f1    = f1_score(y_true, preds, average="weighted", zero_division=0)
            if f1 > best["f1"]:
                best = {"t0": float(round(t0, 2)), "t3": float(round(t3, 2)), "f1": f1}
    print(f"  Optimal thresholds: t0={best['t0']:.2f}, t3={best['t3']:.2f}  "
          f"(val weighted F1={best['f1']:.4f})")
    return best["t0"], best["t3"]


def main():
    np.random.seed(42)
    logger = setup_logging()

    df_eng = load_and_engineer(logger)

    if "Severity" not in df_eng.columns:
        raise KeyError("'Severity' column missing after engineering.")

    y = df_eng["Severity"]

    # Build feature matrix: use TARGET_FEATURES that are present, fill rest with 0
    available = [c for c in TARGET_FEATURES if c in df_eng.columns]
    missing   = [c for c in TARGET_FEATURES if c not in df_eng.columns]
    if missing:
        logger.warning("Missing features (will fill 0): %s", missing)
        for col in missing:
            df_eng[col] = 0

    X = df_eng[TARGET_FEATURES].select_dtypes(include=[np.number]).copy()
    # Re-add any non-numeric that slipped through
    for col in TARGET_FEATURES:
        if col not in X.columns and col in df_eng.columns:
            try:
                X[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)
            except Exception:
                X[col] = 0

    X = X.reindex(columns=TARGET_FEATURES).fillna(0)
    feature_cols = X.columns.tolist()
    logger.info("Features: %d  |  Target classes: %s", len(feature_cols), sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_enc   = label_encoder.fit_transform(y_train)
    y_test_enc    = label_encoder.transform(y_test)
    logger.info("Label mapping: %s", dict(enumerate(label_encoder.classes_)))

    # No class weights — let model fit naturally; threshold grid-search handles rare classes
    class_dist = pd.Series(y_train_enc).value_counts().sort_index()
    logger.info("Train class distribution: %s", class_dist.to_dict())
    cw = None
    logger.info("Class weights: None (natural distribution)")

    import tensorflow as tf

    model = build_model(X_train_sc.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
    ]

    logger.info("Training …")
    model.fit(
        X_train_sc, y_train_enc,
        validation_split=0.2,
        epochs=100,
        batch_size=512,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # Threshold optimisation on the held-out validation split
    val_size  = int(len(X_train_sc) * 0.2)
    X_val_sc  = X_train_sc[-val_size:]
    y_val_enc = y_train_enc[-val_size:]
    val_proba = model.predict(X_val_sc, verbose=0)
    t0, t3    = optimise_thresholds(val_proba, y_val_enc)

    # Evaluate on held-out test set
    y_proba    = model.predict(X_test_sc, verbose=0)
    y_pred_enc = apply_thresholds(y_proba, t0, t3)

    acc = accuracy_score(y_test_enc, y_pred_enc)
    f1w = f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0)

    print("\n=== DNN Evaluation (Test Set) ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (weighted): {f1w:.4f}")
    print(f"Thresholds:    t0={t0:.2f}, t3={t3:.2f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test_enc, y_pred_enc,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0,
    ))

    print("\n=== DNN vs Traditional ML (XGBoost) ===")
    print(f"{'Metric':<22} {'XGBoost':>10} {'DNN':>10}")
    print("-" * 44)
    print(f"{'Accuracy':<22} {'0.8400':>10} {acc:>10.4f}")
    print(f"{'Weighted F1':<22} {'0.8100':>10} {f1w:>10.4f}")

    metrics = {
        "accuracy":        acc,
        "weighted_f1":     f1w,
        "xgb_accuracy":    0.84,
        "xgb_weighted_f1": 0.81,
        "t0": t0,
        "t3": t3,
    }

    logger.info("Saving artifacts to %s", SAVED_MODEL_DIR)
    model.save(SAVED_MODEL_DIR / "model.keras")
    joblib.dump(scaler,        SAVED_MODEL_DIR / "scaler.joblib")
    joblib.dump(label_encoder, SAVED_MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols,  SAVED_MODEL_DIR / "feature_columns.joblib")
    joblib.dump(metrics,       SAVED_MODEL_DIR / "metrics.joblib")
    joblib.dump({"t0": t0, "t3": t3}, SAVED_MODEL_DIR / "thresholds.joblib")
    logger.info("All artifacts saved.")


if __name__ == "__main__":
    main()
