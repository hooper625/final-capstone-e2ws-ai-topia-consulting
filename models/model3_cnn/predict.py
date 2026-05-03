#!/usr/bin/env python3
"""
Model 3: Pothole Detection — Prediction Script
===============================================
Loads the trained EfficientNetB0 model and predicts pothole vs. no_pothole
for images in test_data/.

Artifacts expected in models/model3_cnn/saved_model/:
  model.keras, threshold.joblib, metrics.joblib

To run from project root:
    python -u models/model3_cnn/predict.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
MODEL_DIR       = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"
TEST_DATA_DIR   = PROJECT_ROOT / "test_data"
OUTPUT_FILE     = TEST_DATA_DIR / "model3_results.csv"

IMG_SIZE = 384


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model3_predict")
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


def load_artifacts(logger: logging.Logger):
    required = ["model.keras", "threshold.joblib"]
    missing  = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {MODEL_DIR}: {missing}\n"
            "Run train.py first to generate them."
        )
    logger.info("Loading model from %s", MODEL_DIR)
    model     = tf.keras.models.load_model(MODEL_DIR / "model.keras")
    threshold = joblib.load(MODEL_DIR / "threshold.joblib")
    logger.info("Threshold: %.2f", threshold)
    return model, threshold


def crop_road_region(img):
    h  = tf.shape(img)[0]
    w  = tf.shape(img)[1]
    y1 = tf.cast(tf.cast(h, tf.float32) * 0.30, tf.int32)
    y2 = tf.cast(tf.cast(h, tf.float32) * 0.78, tf.int32)
    x1 = tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32)
    x2 = tf.cast(tf.cast(w, tf.float32) * 0.95, tf.int32)
    return img[y1:y2, x1:x2, :]


def preprocess_image(img_path: str) -> np.ndarray:
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = crop_road_region(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img * 255.0)
    return tf.expand_dims(img, axis=0).numpy()


def predict_batch(image_paths: list, model, threshold: float) -> pd.DataFrame:
    results = []
    for path in image_paths:
        try:
            img_array = preprocess_image(path)
            prob      = float(model.predict(img_array, verbose=0)[0][0])
            label     = "pothole" if prob >= threshold else "no_pothole"
            results.append({"image_id": Path(path).name, "predicted_class": label, "confidence": round(prob, 4)})
        except Exception as e:
            results.append({"image_id": Path(path).name, "predicted_class": "error", "confidence": 0.0})
    return pd.DataFrame(results)


def main():
    logger = setup_logging()
    model, threshold = load_artifacts(logger)

    image_exts = {".jpg", ".jpeg", ".png"}
    image_paths = [
        str(p) for p in TEST_DATA_DIR.iterdir()
        if p.suffix.lower() in image_exts
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_DATA_DIR}")
    logger.info("Found %d images", len(image_paths))

    results = predict_batch(image_paths, model, threshold)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info("Predictions saved to %s", OUTPUT_FILE)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
