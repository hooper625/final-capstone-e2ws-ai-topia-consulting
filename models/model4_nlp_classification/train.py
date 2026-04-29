#!/usr/bin/env python3
"""
Model 4: NLP Classification — SGD Pipeline (Notebook → Production)

Clean training script for UrbanPulse 311 complaint classification.

Design choices for this version:
- Fast and stable training
- Clear step-by-step logging
- Saves only the artifacts needed by predict.py

Approach:

* TF-IDF (word + char)
* SGDClassifier (fast, scalable)

to run, go to project root folder
python -u models/model4_nlp_classification/train.py
"""

from __future__ import annotations

import logging
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Model saving
import os
import joblib

from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model4_nlp_classification/saved_model/")

PROJECT_ROOT = Path.cwd()#.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "urbanpulse_311_complaints.csv"
#OUTPUT_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification"
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
# -----------------------------

# 1. Load Data

# -----------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# -----------------------------

# 2. Text Preprocessing

# -----------------------------

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(series):
    return series.apply(clean_text)


def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def get_top_complaint_types(df: pd.DataFrame, n: int = 5) -> list:
    return df["complaint_type"].value_counts().head(n).index.tolist()


def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map complaint types to the top 5 categories + "Other" (6 classes total).
    """
    top_5 = get_top_complaint_types(df, n=5)
    
    df['complaint_category'] = df['complaint_type'].apply(
        lambda x: x if x in top_5 else 'other'
    )

    print("Complaint category distribution:")
    print(df['complaint_category'].value_counts())

    coverage = df[df['complaint_category'] != 'other'].shape[0] / len(df) * 100
    print(f"\nTop 5 categories cover {coverage:.1f}% of all complaints")
    print(f"Total classes: {df['complaint_category'].nunique()} (top 5 + other)")

    return df



# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model4_train")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------
# Custom transformer for extra keyword features
# =========================================================
class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str).str.lower()

        extra = pd.DataFrame({
        "is_driveway": s.str.contains(r"\bdriveway\b", regex=True).astype(int),
        "is_parking": s.str.contains(r"\bparking\b|\bparked\b|\bvehicle\b|\bcar\b", regex=True).astype(int),
        "is_blocked": s.str.contains(r"\bblocked\b|\bblocking\b", regex=True).astype(int),
        "is_noise": s.str.contains(r"\bbanging\b|\bpounding\b|\bloud\b|\bmusic\b", regex=True).astype(int),
    })

        return csr_matrix(extra.values)
    
    
# -----------------------------

# Main Pipeline

# -----------------------------

def main():
    
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 200)
    np.set_printoptions(suppress=True)
    random.seed(42)
    np.random.seed(42)

    print("Libraries Imported!")

    # -----------------------------

    # Paths

    # -----------------------------



    print("Project root:", PROJECT_ROOT)
    print("Data path:", DATA_PATH)
    print("Output dir:", SAVED_MODEL_DIR)
    
    LOGGER = setup_logging()

    TEXT_COLS = ["descriptor", "resolution_description"]


    print("Loading data...")
    df = load_data()

    df = df.copy()

    for col in TEXT_COLS:
        if col not in df.columns:
            LOGGER.warning("Missing column '%s'; creating empty fallback", col)
            df[col] = ""
        df[col] = df[col].fillna("")

    df = create_complaint_categories(df)

    LOGGER.info("Building complaint_text field")
    df["complaint_text"] = (df["descriptor"].map(clean_text) + " | " + df["resolution_description"].map(clean_text))

    df["categories"] = df["complaint_category"].fillna("").map(clean_text)

    df = df[(df["complaint_text"] != "") & (df["categories"] != "")].copy()
    df = df.reset_index(drop=True)


    LOGGER.info("Finished preprocessing")


    df_model = df.copy()

    df_model["complaint_text"] = df_model["complaint_text"].fillna("").map(clean_text)
    df_model = df_model[(df_model["complaint_text"] != "") & (df_model["categories"].notna())].copy()

    print("Modeling shape:", df_model.shape)
    print(df_model["categories"].value_counts())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_model["categories"])

    X_train_text, X_val_text, y_train, y_val = train_test_split(
    df_model["complaint_text"],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
    )

    LOGGER.info("Train/test split done")


    # =========================================================
    # Full pipeline
    # =========================================================
    model_pipeline = Pipeline([
        (
        "features",
        FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    max_features=12000,
                    min_df=10,
                    max_df=0.95,
                    # stop_words=STOP_WORDS,
                    sublinear_tf=True
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    max_features=5000,
                    min_df=10,
                    max_df=0.95,
                    sublinear_tf=True
                )
            ),
            (
                "extra_features",
                ExtraTextFeatures()
            ),
        ])
    ),
    (
        "clf",
        SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            max_iter=1000,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
            n_jobs=-1
            )
        )
    ])


    # =========================================================
    # 3. Fit / predict
    # =========================================================
    model_pipeline.fit(X_train_text, y_train)

    y_pred = model_pipeline.predict(X_val_text)

    # probabilities / confidence
    y_proba = model_pipeline.predict_proba(X_val_text)
    y_conf = y_proba.max(axis=1)



    acc = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    #print("Accuracy:", round(acc, 4))
    #print("Weighted F1:", round(f1_weighted, 4))

    #print("\nClassification Report:")
    #print(
    #    classification_report(
    #        y_val,
    #        y_pred,
    #        target_names=label_encoder.classes_,
    #        zero_division=0
    #        )
    #    )

    #cm = confusion_matrix(y_val, y_pred)
    #cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    #print("\nConfusion Matrix:")
    #print(cm_df)



    LOGGER.info("Evaluating %s", "Model")

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(
        f"Precision (weighted): "
        f"{precision_score(y_val, y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall (weighted): "
        f"{recall_score(y_val, y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1 (weighted): "
        f"{f1_score(y_val, y_pred, average='weighted', zero_division=0):.4f}"
    )



    LOGGER.info("Complaint Routing Recommendation")
    
    """
    # Build Complaint Routing Recommendations
    
    Use the same Training TEXT to predict which "agency" should the complaint be routed to. 
    Such as:
    
    **"blocked driveway"** --- nypd
    
    **"heat/hot water"** --- hpd
    
    **"illegal parking"** --- nypd
    
    **"noise - residential"** --- nypd
    
    **"snow or ice"** --- dsny
    """

    # -----------------------------
    # Encode labels for route-to-agency
    # -----------------------------
    df_model2 = df_model.copy()

    label_encoder_rt = LabelEncoder()
    df_model2["rt_label"] = label_encoder_rt.fit_transform(df_model2["agency"])

    id2label_rt = {i: label for i, label in enumerate(label_encoder_rt.classes_)}
    label2id_rt = {label: i for i, label in id2label_rt.items()}

    print("\nLabel mapping:")
    print(id2label_rt)


    X_train_text_rt, X_val_text_rt, y_train_rt, y_test_rt = train_test_split(
        df_model2["complaint_text"],
        df_model2["rt_label"],
        test_size=0.2,
        random_state=42,
        stratify=df_model2["rt_label"]
    )


    # =========================================================
    # Full pipeline
    # =========================================================
    model_pipeline_rt = Pipeline([
    (
        "features",
        FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    max_features=12000,
                    min_df=10,
                    max_df=0.95,
                    # stop_words=STOP_WORDS,
                    sublinear_tf=True
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    max_features=5000,
                    min_df=10,
                    max_df=0.95,
                    sublinear_tf=True
                )
            ),
            (
                "extra_features",
                ExtraTextFeatures()
            ),
        ])
    ),
    (
        "clf",
        SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=20,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )
    ])




    # =========================================================
    # Fit / predict
    # =========================================================
    model_pipeline_rt.fit(X_train_text_rt, y_train_rt)

    y_pred_rt = model_pipeline_rt.predict(X_val_text_rt)

    # optional probabilities / confidence
    y_proba_rt = model_pipeline_rt.predict_proba(X_val_text_rt)
    y_conf_rt = y_proba_rt.max(axis=1)


    acc_rt = accuracy_score(y_test_rt, y_pred_rt)
    f1_weighted_rt = f1_score(y_test_rt, y_pred_rt, average="weighted")

    print("Routing Accuracy:", round(acc_rt, 4))
    print("Routing Weighted F1:", round(f1_weighted_rt, 4))

    print("\nRouting Classification Report:")
    print(
    classification_report(
        y_test_rt,
        y_pred_rt,
        target_names=label_encoder_rt.classes_,
        zero_division=0
        )
    )

    ## Save models and artifacts


    joblib.dump(model_pipeline, SAVED_MODEL_DIR / "model4_category_classifier_char_tfidf_SGD.pkl")
    joblib.dump(model_pipeline_rt, SAVED_MODEL_DIR / "model4_routing_classifier_char_tfidf_SGD.pkl")
    joblib.dump(label_encoder, SAVED_MODEL_DIR / "model4_category_label_encoder.pkl")
    joblib.dump(label_encoder_rt, SAVED_MODEL_DIR / "model4_routing_label_encoder.pkl")

    print("Saved artifacts of recommendation router to:", SAVED_MODEL_DIR)



if __name__ == "__main__":
    main()