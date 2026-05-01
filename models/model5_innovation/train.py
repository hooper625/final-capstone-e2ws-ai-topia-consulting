#!/usr/bin/env python3
"""
Model 5: Resolution Outcome Predictor (NLP)
============================================
Predicts complaint resolution outcome and estimated resolution time
from complaint text and metadata — using the resolution_description
field as the label source.

Value proposition: Tells residents and dispatchers whether a complaint
is likely to be Resolved, Unresolved, or Referred before it is processed,
enabling smarter prioritisation and expectation-setting.
"""
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

SAVE_DIR  = Path("models/model5_innovation/saved_model/")
DATA_PATH = Path("data/raw/urbanpulse_311_complaints.csv")

# ── City / location normalisation ────────────────────────────────────────────
CITY_SUBS = [
    (r"New York City Police Department", "Nova Haven Police Department"),
    (r"New York City",                   "Nova Haven"),
    (r"City of New York",                "Nova Haven"),
    (r"New Yorkers?",                    "Nova Haven residents"),
    (r"NYC Parks",                       "Nova Haven Parks"),
    (r"\bNYC\b",                         "Nova Haven"),
    (r"\bNYPD\b",                        "NHPD"),
    (r"\bN\.Y\.P\.D\.?\b",              "NHPD"),
    (r"\bManhattan\b",                   "Central District"),
    (r"\bBrooklyn\b",                    "East District"),
    (r"\bBronx\b",                       "North District"),
    (r"\bQueens\b",                      "West District"),
    (r"Staten Island",                   "South District"),
    (r"HPDONLINE",                       "the housing portal"),
]

AGENCY_REMAP = {"NYPD": "NPD"}


def strip_city_refs(text: str) -> str:
    if pd.isna(text):
        return ""
    for pattern, replacement in CITY_SUBS:
        text = re.sub(pattern, replacement, str(text), flags=re.IGNORECASE)
    return text.strip()


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_outcome(text: str) -> str:
    """3-class label from resolution description: Resolved | Unresolved | Referred"""
    t = str(text).lower()

    if re.search(
        r"has been referred to|referred to the|forwarded to"
        r"|not within its jurisdiction|redirected to",
        t,
    ):
        return "Referred"

    if re.search(
        r"no criminal violation|no violation|could not|unable to"
        r"|not able to|did not find|no evidence|not observe"
        r"|could not be verified|was not able|no access|not accessible"
        r"|officers were unable|did not violate|not violate",
        t,
    ):
        return "Unresolved"

    return "Resolved"


def time_bucket(days: float) -> str:
    if days < 1:
        return "Same Day"
    elif days <= 7:
        return "1–7 Days"
    else:
        return "8+ Days"


def main():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH, usecols=[
        "created_date", "closed_date", "agency",
        "complaint_type", "descriptor", "resolution_description",
        "borough", "open_data_channel_type", "status",
    ])
    print(f"  Total rows: {len(df):,}")

    df["created_date"]  = pd.to_datetime(df["created_date"],  errors="coerce")
    df["closed_date"]   = pd.to_datetime(df["closed_date"],   errors="coerce")
    df["days_to_close"] = (
        (df["closed_date"] - df["created_date"]).dt.total_seconds() / 86400
    )
    df["hour"]        = df["created_date"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["created_date"].dt.dayofweek.fillna(0).astype(int)
    df["month"]       = df["created_date"].dt.month.fillna(1).astype(int)

    df["agency"] = df["agency"].map(lambda x: AGENCY_REMAP.get(x, x))

    df = df[
        (df["status"] == "Closed")
        & (df["days_to_close"] >= 0)
        & (df["days_to_close"] <= 365)
    ].copy().reset_index(drop=True)
    print(f"  Closed with valid time: {len(df):,}")

    df["res_clean"] = df["resolution_description"].apply(strip_city_refs)
    df["outcome"]   = df["res_clean"].apply(classify_outcome)
    df["time_lbl"]  = df["days_to_close"].apply(time_bucket)

    print("\nOutcome distribution:")
    print(df["outcome"].value_counts().to_string())
    print("\nTime-bucket distribution:")
    print(df["time_lbl"].value_counts().to_string())

    df["input_text"] = (
        df["complaint_type"].fillna("") + " " + df["descriptor"].fillna("")
    ).apply(clean_text)

    df["borough"]                = df["borough"].fillna("Unspecified")
    df["open_data_channel_type"] = df["open_data_channel_type"].fillna("UNKNOWN")

    X_text    = df["input_text"].values
    X_cat     = df[["agency", "borough", "open_data_channel_type"]].values
    X_num     = df[["hour", "day_of_week", "month"]].values.astype(float)

    ord_enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_enc = ord_enc.fit_transform(X_cat)
    X_numeric = np.hstack([X_cat_enc, X_num])

    tfidf  = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        max_features=12000, sublinear_tf=True, min_df=2,
    )
    X_tfidf = tfidf.fit_transform(X_text)
    X_full  = hstack([X_tfidf, csr_matrix(X_numeric)])

    # Outcome classifier
    outcome_le = LabelEncoder()
    y_outcome  = outcome_le.fit_transform(df["outcome"].values)
    Xtr, Xte, ytr, yte = train_test_split(
        X_full, y_outcome, test_size=0.15, random_state=42, stratify=y_outcome,
    )
    print("\nTraining outcome classifier …")
    outcome_clf = LogisticRegression(max_iter=600, C=1.0, solver="lbfgs")
    outcome_clf.fit(Xtr, ytr)
    ypred      = outcome_clf.predict(Xte)
    outcome_f1 = f1_score(yte, ypred, average="weighted")
    print(f"  Outcome weighted F1: {outcome_f1:.4f}")
    print(classification_report(yte, ypred, target_names=outcome_le.classes_))

    # Time classifier
    time_le = LabelEncoder()
    y_time  = time_le.fit_transform(df["time_lbl"].values)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_full, y_time, test_size=0.15, random_state=42, stratify=y_time,
    )
    print("Training time classifier …")
    time_clf = LogisticRegression(max_iter=600, C=1.0, solver="lbfgs")
    time_clf.fit(Xtr2, ytr2)
    ypred2  = time_clf.predict(Xte2)
    time_f1 = f1_score(yte2, ypred2, average="weighted")
    print(f"  Time weighted F1:    {time_f1:.4f}")
    print(classification_report(yte2, ypred2, target_names=time_le.classes_))

    # Save all artifacts
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(outcome_clf, SAVE_DIR / "outcome_clf.joblib")
    joblib.dump(time_clf,    SAVE_DIR / "time_clf.joblib")
    joblib.dump(tfidf,       SAVE_DIR / "tfidf.joblib")
    joblib.dump(ord_enc,     SAVE_DIR / "ord_enc.joblib")
    joblib.dump(outcome_le,  SAVE_DIR / "outcome_le.joblib")
    joblib.dump(time_le,     SAVE_DIR / "time_le.joblib")
    joblib.dump({
        "outcome_f1":          outcome_f1,
        "time_f1":             time_f1,
        "n_train":             len(df),
        "outcome_classes":     list(outcome_le.classes_),
        "time_classes":        list(time_le.classes_),
        "ord_enc_categories":  [list(c) for c in ord_enc.categories_],
    }, SAVE_DIR / "metrics.joblib")
    print(f"\nAll artifacts saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
