#!/usr/bin/env python3
"""
Model 5: Innovation — Training Script
=====================================

Urban Complaint Response Optimizer (Unsupervised / Hybrid Scoring)

Updated fix:
- KMeans cluster IDs are NOT business labels. Cluster 0 does not automatically mean urgent.
- This version evaluates smoke tests by urgency_score / urgency_tier, not raw cluster_id.
- Adds stronger urgency scoring for:
  * gas leak -> urgent
  * heat/hot water -> urgent/elevated
  * snow or ice -> elevated
  * blocked driveway -> elevated
  * illegal parking -> elevated
  * noise - residential -> elevated
- Keeps clustering unsupervised, but the operational output is the hybrid urgency tier.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "urbanpulse_311_complaints.csv"
RULE_TRAINING_PATH = PROCESSED_DATA / "model4_agency_rule_training_examples_1000.csv"
RAW_DATA = PROJECT_ROOT / "data" / "raw"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TEXT_COLS = ["complaint_type", "descriptor", "resolution_description"]
PRIMARY_ID_COL = "unique_key"
DEFAULT_N_CLUSTERS = 3
TFIDF_MAX_FEATURES = 8000

MIN_AGENCY_COUNT_FOR_TRAINING = 100
SYNTHETIC_OVERSAMPLE_TIMES = 10


# ---------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------
URGENT_PATTERNS = [
    r"\bgas leak\b", r"\bsmell of gas\b", r"\bstrong smell of gas\b",
    r"\bfire\b", r"\bflood\b", r"\bflooding\b", r"\bleak\b",
    r"\bsmoke\b", r"\bexplosion\b", r"\bunsafe\b", r"\bdanger\b", r"\bhazard\b",
    r"\bhazardous\b", r"\binjury\b", r"\binjured\b", r"\bblood\b",
    r"\baccident\b", r"\bcrash\b", r"\bcollapse\b", r"\bsinkhole\b",
    r"\bno heat\b", r"\bno hot water\b", r"\bno heat and no hot water\b",
    r"\bno water\b", r"\bsewage\b", r"\blive wire\b", r"\belectrical\b",
    r"\bpower outage\b", r"\bemergency\b", r"\burgent\b", r"\bimmediately\b",
    r"\basap\b", r"\bchild\b", r"\belderly\b", r"\bdisabled\b",
    r"\bwheelchair\b", r"\bmedical\b", r"\bambulance\b", r"\b911\b",
    r"\bviolence\b", r"\bassault\b", r"\bthreat\b", r"\bweapon\b",
]

DISTRESS_PATTERNS = [
    r"\bhelp\b", r"\bplease help\b", r"\bdesperate\b", r"\bcrying\b",
    r"\bscared\b", r"\bterrified\b", r"\bcannot breathe\b", r"\bpanic\b",
    r"\bstuck\b", r"\btrapped\b", r"\bstranded\b", r"\bno response\b",
    r"\bwaiting for hours\b", r"\bkids\b", r"\bbaby\b", r"\bgrandma\b",
    r"\bgrandfather\b", r"\bsick\b", r"\bunsafe for family\b",
]

MODERATE_PATTERNS = [
    r"\brepair\b", r"\bbroken\b", r"\bcracked\b", r"\bpothole\b",
    r"\btrash\b", r"\bgarbage\b", r"\bnoise\b", r"\bparking\b",
    r"\bstreet light\b", r"\blight out\b", r"\bsign missing\b",
    r"\bwater leak\b", r"\bmold\b", r"\bdrain\b", r"\bstanding water\b",
    r"\broad damage\b", r"\bsidewalk\b", r"\btraffic signal\b",
    r"\bmissed pickup\b", r"\brodent\b", r"\binfestation\b",
    r"\bsnow\b", r"\bice\b", r"\bicy\b", r"\bdriveway\b",
    r"\bhot water\b", r"\bblocked driveway\b", r"\billegal parking\b",
    r"\bnoise - residential\b", r"\bdouble parked\b", r"\bhydrant\b",
]

DSNY_PATTERNS = [
    r"\bsnow\b", r"\bice\b", r"\bicy\b", r"\bslush\b",
    r"\btrash\b", r"\bgarbage\b", r"\brecycling\b", r"\bsanitation\b",
    r"\bmissed pickup\b", r"\bmissed collection\b",
]
NYPD_PATTERNS = [
    r"\bblocked driveway\b", r"\bdriveway\b", r"\billegal parking\b",
    r"\bdouble parked\b", r"\bhydrant\b", r"\bnoise\b", r"\bloud\b",
    r"\bmusic\b", r"\bbanging\b", r"\bparty\b",
]
HPD_PATTERNS = [
    r"\bheat/hot water\b", r"\bno heat\b", r"\bno hot water\b",
    r"\bhot water\b", r"\bheating\b", r"\bboiler\b", r"\bradiator\b",
    r"\bapartment\b", r"\blandlord\b", r"\btenant\b",
]
DOB_PATTERNS = [
    r"\bconstruction\b", r"\bpermit\b", r"\bscaffold\b",
    r"\bdemolition\b", r"\bunsafe construction\b",
]
OOS_PATTERNS = [
    r"\bsheriff\b", r"\bmarshal\b", r"\beviction\b", r"\blockout\b",
    r"\bcivil enforcement\b",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model5_innovation_train")
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


LOGGER = setup_logging()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("drive-way", "driveway")
    text = text.replace("drive way", "driveway")
    text = text.replace("hotwater", "hot water")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_complaint_type(x: str) -> str:
    x = clean_text(x)
    aliases = {
        "blocked drive way": "blocked driveway",
        "blocked drive-way": "blocked driveway",
        "heat hot water": "heat/hot water",
        "heat and hot water": "heat/hot water",
        "heating hot water": "heat/hot water",
        "noise residential": "noise - residential",
        "residential noise": "noise - residential",
        "snow ice": "snow or ice",
        "snow and ice": "snow or ice",
    }
    return aliases.get(x, x)


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 0
    text = text.lower()
    return sum(int(bool(re.search(pattern, text))) for pattern in patterns)


def safe_json(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def find_input_csv() -> Path:
    return DATA_PATH


def build_proxy_priority(row: pd.Series) -> Tuple[int, str]:
    if row["urgent_keyword_count"] > 0 or row["distress_keyword_count"] > 0:
        return 1, "urgent"
    if row["moderate_keyword_count"] > 0:
        return 2, "elevated"
    return 3, "normal"


def assign_operational_tier(row: pd.Series) -> str:
    """
    Deterministic scoring tier. This is not hard-coded agency routing.
    It is urgency/priority scoring for Model 5.
    """
    if row["urgent_keyword_count"] > 0 or row["distress_keyword_count"] > 0:
        return "urgent"

    # Operationally important city-service complaints should not be normal.
    if (
        row["dsny_signal_count"] > 0
        or row["nypd_signal_count"] > 0
        or row["hpd_signal_count"] > 0
        or row["moderate_keyword_count"] > 0
    ):
        return "elevated"

    if row["urgency_score"] >= 0.30:
        return "urgent"
    if row["urgency_score"] >= 0.05:
        return "elevated"
    return "normal"


def compute_score_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["urgent_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, URGENT_PATTERNS)
    )
    out["distress_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, DISTRESS_PATTERNS)
    )
    out["moderate_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, MODERATE_PATTERNS)
    )

    out["dsny_signal_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, DSNY_PATTERNS)
    )
    out["nypd_signal_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, NYPD_PATTERNS)
    )
    out["hpd_signal_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, HPD_PATTERNS)
    )
    out["dob_signal_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, DOB_PATTERNS)
    )
    out["oos_signal_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, OOS_PATTERNS)
    )

    out["text_length"] = out["complaint_text"].str.len().fillna(0)
    out["exclamation_count"] = out["complaint_text"].str.count("!").fillna(0)
    out["all_caps_word_count"] = out["complaint_text"].apply(
        lambda x: len(re.findall(r"\b[A-Z]{3,}\b", x)) if isinstance(x, str) else 0
    )

    priority_info = out.apply(build_proxy_priority, axis=1, result_type="expand")
    out["proxy_priority"] = priority_info[0]
    out["proxy_tier"] = priority_info[1]
    return out


# ---------------------------------------------------------------------
# Main pipeline functions
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    data_path = find_input_csv()
    LOGGER.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)
    LOGGER.info("Loaded %s rows and %s columns", df.shape[0], df.shape[1])

    required_cols = ["complaint_type", "descriptor", "resolution_description", "agency"]
    for col in required_cols:
        if col not in df.columns:
            LOGGER.warning("Missing column '%s'; creating empty fallback", col)
            df[col] = ""

    if PRIMARY_ID_COL not in df.columns:
        LOGGER.warning("Missing id column '%s'; generating synthetic ids", PRIMARY_ID_COL)
        df[PRIMARY_ID_COL] = np.arange(1, len(df) + 1)

    df["source"] = "raw"

    if RULE_TRAINING_PATH.exists():
        df_rules = pd.read_csv(RULE_TRAINING_PATH)
        for col in required_cols:
            if col not in df_rules.columns:
                df_rules[col] = ""

        df_rules = df_rules[required_cols].copy()
        df_rules[PRIMARY_ID_COL] = np.arange(
            int(df[PRIMARY_ID_COL].max()) + 1,
            int(df[PRIMARY_ID_COL].max()) + 1 + len(df_rules),
        )
        df_rules["source"] = "synthetic_rule"
        df_rules["complaint_type"] = df_rules["complaint_type"].map(normalize_complaint_type)
        df_rules["agency"] = df_rules["agency"].fillna("").astype(str).str.upper().str.strip()

        df_rules_os = pd.concat([df_rules] * SYNTHETIC_OVERSAMPLE_TIMES, ignore_index=True)
        df = pd.concat([df, df_rules_os], ignore_index=True)

        LOGGER.info(
            "Added synthetic rule rows: %s x %s = %s",
            len(df_rules),
            SYNTHETIC_OVERSAMPLE_TIMES,
            len(df_rules_os),
        )
        LOGGER.info("Synthetic agency distribution:\n%s", df_rules["agency"].value_counts())
        LOGGER.info("Synthetic complaint type distribution:\n%s", df_rules["complaint_type"].value_counts())
    else:
        LOGGER.warning("Synthetic rule-training file not found, skipping: %s", RULE_TRAINING_PATH)

    if "agency" in df.columns:
        df["agency"] = df["agency"].fillna("").astype(str).str.upper().str.strip()
        agency_counts = df["agency"].value_counts()
        rare_agencies = agency_counts[agency_counts < MIN_AGENCY_COUNT_FOR_TRAINING]

        if not rare_agencies.empty:
            LOGGER.warning(
                "Dropping rare agencies from Model 5 training because they have fewer than %s rows:\n%s",
                MIN_AGENCY_COUNT_FOR_TRAINING,
                rare_agencies,
            )
            df = df[~df["agency"].isin(rare_agencies.index)].copy()

        LOGGER.info("Agency distribution after rare-agency filter:\n%s", df["agency"].value_counts())

    return df


def preprocess(df: pd.DataFrame):
    work = df.copy()

    for col in TEXT_COLS:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("")

    work["complaint_type"] = work["complaint_type"].map(normalize_complaint_type)

    LOGGER.info("Building complaint_text field")
    work["complaint_text"] = (
        work["complaint_type"].map(clean_text)
        + " | "
        + work["descriptor"].map(clean_text)
        + " | "
        + work["resolution_description"].map(clean_text)
    ).str.strip(" |")

    work = work[work["complaint_text"].str.strip() != ""].copy()
    work = work.reset_index(drop=True)

    work = compute_score_features(work)

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(work["complaint_text"])
    LOGGER.info("TF-IDF matrix shape: %s", X_tfidf.shape)

    return work, tfidf, X_tfidf


def train_model(df: pd.DataFrame, X_tfidf):
    LOGGER.info("Training KMeans clustering model with %s clusters", DEFAULT_N_CLUSTERS)
    kmeans = KMeans(
        n_clusters=DEFAULT_N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(X_tfidf)

    cluster_profile = (
        df.groupby("cluster_id")[
            [
                "urgent_keyword_count",
                "distress_keyword_count",
                "moderate_keyword_count",
                "dsny_signal_count",
                "nypd_signal_count",
                "hpd_signal_count",
                "oos_signal_count",
            ]
        ]
        .mean()
        .reset_index()
    )

    cluster_profile["cluster_severity_score"] = (
        2.0 * cluster_profile["urgent_keyword_count"]
        + 1.5 * cluster_profile["distress_keyword_count"]
        + 1.0 * cluster_profile["moderate_keyword_count"]
        + 0.25 * cluster_profile["dsny_signal_count"]
        + 0.25 * cluster_profile["nypd_signal_count"]
        + 0.25 * cluster_profile["hpd_signal_count"]
        - 0.25 * cluster_profile["oos_signal_count"]
    )
    cluster_profile = cluster_profile.sort_values("cluster_severity_score").reset_index(drop=True)

    cluster_rank_map = {
        safe_json(cluster_profile.loc[i, "cluster_id"]): i
        for i in range(len(cluster_profile))
    }
    df["cluster_severity_rank"] = df["cluster_id"].map(cluster_rank_map)

    score_cols = [
        "urgent_keyword_count",
        "moderate_keyword_count",
        "distress_keyword_count",
        "dsny_signal_count",
        "nypd_signal_count",
        "hpd_signal_count",
        "text_length",
        "exclamation_count",
        "all_caps_word_count",
        "cluster_severity_rank",
    ]

    scaler = MinMaxScaler()
    scaled_cols = [f"{c}_scaled" for c in score_cols]
    df[scaled_cols] = scaler.fit_transform(df[score_cols])

    df["urgency_score"] = (
        0.36 * df["urgent_keyword_count_scaled"]
        + 0.12 * df["moderate_keyword_count_scaled"]
        + 0.24 * df["distress_keyword_count_scaled"]
        + 0.10 * df["cluster_severity_rank_scaled"]
        + 0.04 * df["dsny_signal_count_scaled"]
        + 0.04 * df["nypd_signal_count_scaled"]
        + 0.04 * df["hpd_signal_count_scaled"]
        + 0.03 * df["exclamation_count_scaled"]
        + 0.03 * df["all_caps_word_count_scaled"]
    )

    df["urgency_score"] = np.where(
        df["urgent_keyword_count"] > 0,
        df["urgency_score"] + 0.20,
        df["urgency_score"],
    ).clip(0, 1)

    df["urgency_tier"] = df.apply(assign_operational_tier, axis=1)

    tier_priority_map = {"urgent": 1, "elevated": 2, "normal": 3}
    df["response_priority"] = df["urgency_tier"].map(tier_priority_map).astype(int)

    model_bundle = {
        "tfidf_vectorizer": None,
        "kmeans_model": kmeans,
        "scaler": scaler,
        "score_columns": score_cols,
        "scaled_columns": scaled_cols,
        "cluster_profile": cluster_profile.to_dict(orient="records"),
        "cluster_rank_map": {int(k): int(v) for k, v in cluster_rank_map.items()},
        "tier_priority_map": tier_priority_map,
        "config": {
            "n_clusters": DEFAULT_N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "text_columns": TEXT_COLS,
            "id_column": PRIMARY_ID_COL,
            "min_agency_count_for_training": MIN_AGENCY_COUNT_FOR_TRAINING,
            "synthetic_oversample_times": SYNTHETIC_OVERSAMPLE_TIMES,
            "note": "Use urgency_tier / response_priority as operational output. Raw cluster_id is not a severity label.",
        },
    }

    return df, model_bundle


def score_text_for_smoke_test(text: str, model_bundle: Dict) -> Dict:
    """
    Score one text using the same hybrid scoring logic.
    This prevents the misleading pattern of judging Model 5 by raw cluster_id.
    """
    complaint_text = clean_text(text)
    tmp = pd.DataFrame({"complaint_text": [complaint_text]})
    tmp = compute_score_features(tmp)

    tfidf = model_bundle["tfidf_vectorizer"]
    kmeans = model_bundle["kmeans_model"]
    scaler = model_bundle["scaler"]
    score_cols = model_bundle["score_columns"]

    X = tfidf.transform(tmp["complaint_text"])
    cluster_id = int(kmeans.predict(X)[0])
    cluster_rank = int(model_bundle["cluster_rank_map"].get(cluster_id, 0))
    tmp["cluster_id"] = cluster_id
    tmp["cluster_severity_rank"] = cluster_rank

    scaled = scaler.transform(tmp[score_cols])
    scaled_cols = [f"{c}_scaled" for c in score_cols]
    for i, col in enumerate(scaled_cols):
        tmp[col] = scaled[:, i]

    tmp["urgency_score"] = (
        0.36 * tmp["urgent_keyword_count_scaled"]
        + 0.12 * tmp["moderate_keyword_count_scaled"]
        + 0.24 * tmp["distress_keyword_count_scaled"]
        + 0.10 * tmp["cluster_severity_rank_scaled"]
        + 0.04 * tmp["dsny_signal_count_scaled"]
        + 0.04 * tmp["nypd_signal_count_scaled"]
        + 0.04 * tmp["hpd_signal_count_scaled"]
        + 0.03 * tmp["exclamation_count_scaled"]
        + 0.03 * tmp["all_caps_word_count_scaled"]
    )

    tmp["urgency_score"] = np.where(
        tmp["urgent_keyword_count"] > 0,
        tmp["urgency_score"] + 0.20,
        tmp["urgency_score"],
    ).clip(0, 1)

    tmp["urgency_tier"] = tmp.apply(assign_operational_tier, axis=1)
    tmp["response_priority"] = tmp["urgency_tier"].map(model_bundle["tier_priority_map"]).astype(int)

    return tmp.iloc[0].to_dict()


def evaluate_model(df: pd.DataFrame, X_tfidf, kmeans) -> Dict:
    LOGGER.info("Evaluating unsupervised model")

    labels = df["cluster_id"].values
    n_rows = len(df)

    if len(np.unique(labels)) > 1 and n_rows > len(np.unique(labels)):
        sil = float(silhouette_score(X_tfidf, labels, sample_size=min(5000, n_rows), random_state=RANDOM_STATE))

        sample_n = min(10000, n_rows)
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(n_rows, size=sample_n, replace=False)
        dense_X_sample = X_tfidf[sample_idx].toarray()
        labels_sample = labels[sample_idx]

        ch = float(calinski_harabasz_score(dense_X_sample, labels_sample))
        db = float(davies_bouldin_score(dense_X_sample, labels_sample))
    else:
        sil, ch, db = None, None, None

    proxy_code_map = {"normal": 0, "elevated": 1, "urgent": 2}
    proxy_codes = df["proxy_tier"].map(proxy_code_map).values
    pred_codes = df["urgency_tier"].map(proxy_code_map).values

    tier_match_rate = float(np.mean(proxy_codes == pred_codes))
    nmi_proxy = float(normalized_mutual_info_score(proxy_codes, labels))
    ari_proxy = float(adjusted_rand_score(proxy_codes, labels))

    mean_scores = df.groupby("urgency_tier")["urgency_score"].mean().to_dict()
    tier_distribution = df["urgency_tier"].value_counts(normalize=True).sort_index().to_dict()
    cluster_distribution = df["cluster_id"].value_counts(normalize=True).sort_index().to_dict()

    urgent_mean = float(mean_scores.get("urgent", 0.0))
    normal_mean = float(mean_scores.get("normal", 0.0))
    elevated_mean = float(mean_scores.get("elevated", 0.0))

    priority_lift_vs_normal = urgent_mean / normal_mean if normal_mean and normal_mean > 0 else None
    elevated_lift_vs_normal = elevated_mean / normal_mean if normal_mean and normal_mean > 0 else None

    top_decile = max(1, int(0.10 * len(df)))
    top10 = df.nlargest(top_decile, "urgency_score")
    urgent_capture_top10 = float((top10["proxy_tier"] == "urgent").mean())
    baseline_urgent_rate = float((df["proxy_tier"] == "urgent").mean())
    urgent_capture_lift_top10 = (
        urgent_capture_top10 / baseline_urgent_rate if baseline_urgent_rate > 0 else None
    )

    metrics = {
        "model_type": "unsupervised_hybrid_tfidf_kmeans_scoring",
        "n_rows_used_for_training": int(len(df)),
        "n_clusters": int(kmeans.n_clusters),
        "silhouette_score": sil,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
        "proxy_tier_match_rate": tier_match_rate,
        "normalized_mutual_info_vs_proxy_tiers": nmi_proxy,
        "adjusted_rand_index_vs_proxy_tiers": ari_proxy,
        "tier_distribution": {str(k): float(v) for k, v in tier_distribution.items()},
        "cluster_distribution": {str(k): float(v) for k, v in cluster_distribution.items()},
        "business_impact": {
            "baseline_urgent_rate": baseline_urgent_rate,
            "urgent_rate_in_top_10pct_scored_cases": urgent_capture_top10,
            "top_10pct_urgent_capture_lift": urgent_capture_lift_top10,
            "mean_urgency_score_by_tier": {k: float(v) for k, v in mean_scores.items()},
            "priority_lift_vs_normal": priority_lift_vs_normal,
            "elevated_lift_vs_normal": elevated_lift_vs_normal,
        },
        "important_note": "Raw cluster_id is arbitrary and should not be interpreted as urgent/elevated/normal. Use urgency_tier or response_priority.",
    }

    LOGGER.info("Silhouette score: %s", metrics["silhouette_score"])
    LOGGER.info("Proxy tier match rate: %.4f", metrics["proxy_tier_match_rate"])
    LOGGER.info("Top 10%% urgent capture lift: %s", metrics["business_impact"]["top_10pct_urgent_capture_lift"])

    return metrics


def run_smoke_tests(model_bundle: Dict) -> None:
    expected = {
        "snow or ice | snow and ice on the road": ["elevated", "urgent"],
        "blocked driveway | someone blocked my driveway": ["elevated", "urgent"],
        "heat/hot water | no heat and no hot water in my apartment": ["urgent"],
        "illegal parking | car parked illegally in front of my house": ["elevated", "urgent"],
        "noise - residential | loud music from neighbor apartment": ["elevated", "urgent"],
        "gas leak and strong smell in building": ["urgent"],
    }

    LOGGER.info("Model 5 operational smoke tests:")
    passed = 0
    for text, allowed_tiers in expected.items():
        scored = score_text_for_smoke_test(text, model_bundle)
        tier = scored["urgency_tier"]
        ok = tier in allowed_tiers
        passed += int(ok)
        LOGGER.info(
            "%s | text=%r | cluster=%s | cluster_rank=%s | urgency_score=%.4f | urgency_tier=%s | expected=%s",
            "PASS" if ok else "FAIL",
            text,
            scored["cluster_id"],
            scored["cluster_severity_rank"],
            scored["urgency_score"],
            tier,
            allowed_tiers,
        )

    LOGGER.info("Model 5 smoke test pass rate: %s/%s", passed, len(expected))


def save_model(model_bundle: Dict, metrics: Dict, scored_df: pd.DataFrame) -> None:
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODEL_DIR / "model.joblib"
    metrics_path = SAVED_MODEL_DIR / "metrics.json"
    sample_path = SAVED_MODEL_DIR / "training_scored_sample.csv"

    joblib.dump(model_bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=safe_json)

    cols_to_save = [
        PRIMARY_ID_COL,
        "source",
        "agency",
        "complaint_type",
        "complaint_text",
        "cluster_id",
        "cluster_severity_rank",
        "proxy_tier",
        "urgency_tier",
        "response_priority",
        "urgency_score",
    ]
    available_cols = [c for c in cols_to_save if c in scored_df.columns]
    scored_df[available_cols].head(5000).to_csv(sample_path, index=False)

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved scored sample to %s", sample_path)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Data path: %s", DATA_PATH)
    LOGGER.info("Rule training path: %s", RULE_TRAINING_PATH)
    LOGGER.info("Saved model dir: %s", SAVED_MODEL_DIR)

    df = load_data()

    processed_df, tfidf_vectorizer, X_tfidf = preprocess(df)

    scored_df, model_bundle = train_model(processed_df, X_tfidf)
    model_bundle["tfidf_vectorizer"] = tfidf_vectorizer

    metrics = evaluate_model(
        scored_df,
        X_tfidf,
        model_bundle["kmeans_model"],
    )

    run_smoke_tests(model_bundle)

    save_model(model_bundle, metrics, scored_df)

    LOGGER.info("Training complete!")


if __name__ == "__main__":
    main()
