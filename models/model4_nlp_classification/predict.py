#!/usr/bin/env python3
"""
Dual Prediction Script: Complaint Category + Routing Agency
===========================================================
Loads two saved sklearn pipeline models and predicts:
1) complaint category
2) recommended agency to route to

Important:
This script includes the custom ExtraTextFeatures transformer so that
joblib can unpickle pipelines that were trained with it.

Saved artifacts expected:
- /mnt/data/model4_complaint_classifier_char_tfidf_SGD.pkl
- /mnt/data/model4_category_label_encoder.pkl
- /mnt/data/model4_routing_classifier_char_tfidf_SGD.pkl
- /mnt/data/model4_routing_label_encoder.pkl


to run, go to project root folder
python -u models/model4_nlp_classification/predict.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model4_dual_results.csv"

#MODEL_DIR = SCRIPT_DIR / "saved_model"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
RAW_DATA = PROJECT_ROOT / "data" / "raw"



MODEL_PATH = Path("models/model4_nlp_classification/saved_model/")
#TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"

BASE_DIR = MODEL_PATH
CATEGORY_MODEL_FILE = BASE_DIR / "model4_category_classifier_char_tfidf_SGD.pkl"
CATEGORY_LABEL_FILE = BASE_DIR / "model4_category_label_encoder.pkl"
ROUTING_MODEL_FILE = BASE_DIR / "model4_routing_classifier_char_tfidf_SGD.pkl"
ROUTING_LABEL_FILE = BASE_DIR / "model4_routing_label_encoder.pkl"

TEST_DATA_DIR = Path("test_data")
OUTPUT_FILE = TEST_DATA_DIR / "model4_dual_results.csv"

PROCESSED_DATA = Path("data/processed")
RAW_DATA = Path("data/raw")

MARIAN_MODEL_MAP = {"es": "Helsinki-NLP/opus-mt-es-en"}
SPANISH_MARKERS = [
    "calle", "carretera", "avenida", "autopista", "ruido", "agua", "calor",
    "nieve", "hielo", "bloqueado", "basura", "vehiculo", "vehículo",
    "estacionamiento", "ilegal", "caliente", "frio", "frío", "apartamento",
    "edificio", "queja", "residencial", "entrada", "conducir", "acera",
    "saneamiento", "barrida", "barrer", "denuncia", "musica", "música",
    "sin calor", "sin agua", "agua caliente", "no barrida",
]

_torch = None
_tqdm = None
_MarianMTModel = None
_MarianTokenizer = None
_DEVICE = "cpu"
_TRANSLATION_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}


class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    """
    Must match the custom transformer used in the latest train_fix_no_rare_class_overweight.py.

    IMPORTANT:
    The number and order of columns returned here must match training exactly.
    If this differs, sklearn will raise:
    ValueError: X has #### features, but SGDClassifier is expecting #### features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str).str.lower()

        extra = pd.DataFrame({
            # DSNY
            "has_snow_or_ice": s.str.contains(
                r"\bsnow\b|\bice\b|\bicy\b|\bslush\b",
                regex=True
            ).astype(int),
            "has_sanitation": s.str.contains(
                r"\btrash\b|\bgarbage\b|\brecycling\b|\bsanitation\b|\bdumping\b",
                regex=True
            ).astype(int),

            # NYPD
            "has_driveway": s.str.contains(
                r"\bdriveway\b",
                regex=True
            ).astype(int),
            "has_blocked": s.str.contains(
                r"\bblocked\b|\bblocking\b|\bobstructing\b",
                regex=True
            ).astype(int),
            "has_parking": s.str.contains(
                r"\bparking\b|\bparked\b|\bvehicle\b|\bcar\b|\btruck\b|\bdouble parked\b|\bhydrant\b",
                regex=True
            ).astype(int),
            "has_noise": s.str.contains(
                r"\bnoise\b|\bloud\b|\bmusic\b|\bbanging\b|\byelling\b|\bparty\b",
                regex=True
            ).astype(int),

            # HPD
            "has_heat_hot_water": s.str.contains(
                r"\bno heat\b|\bheat\b|\bheating\b|\bhot water\b|\bboiler\b|\bradiator\b",
                regex=True
            ).astype(int),
            "has_housing": s.str.contains(
                r"\bapartment\b|\btenant\b|\blandlord\b|\bresidential\b",
                regex=True
            ).astype(int),

            # DOB should require true construction/building-safety language
            "has_dob_signal": s.str.contains(
                r"\bconstruction\b|\bpermit\b|\bscaffold\b|\bdemolition\b|\bunsafe construction\b",
                regex=True
            ).astype(int),

            # OOS should not fire unless explicit sheriff / marshal / eviction terms exist
            "has_oos_signal": s.str.contains(
                r"\bsheriff\b|\bmarshal\b|\beviction\b|\blockout\b|\bcivil enforcement\b",
                regex=True
            ).astype(int),
        })

        return csr_matrix(extra.values)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model4_dual_predict")
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


def infer_complaint_type_from_free_text(text: str) -> str:
    """
    Helps prediction text match the enhanced training format when the test file
    has only free-text descriptor / resolution_description.
    """
    t = clean_text(text)

    if re.search(r"\bsnow\b|\bice\b|\bicy\b", t):
        return "snow or ice"
    if re.search(r"\bdriveway\b", t) and re.search(r"\bblock|\bobstruct|park", t):
        return "blocked driveway"
    if re.search(r"\billegal parking\b|\bdouble parked\b|\bno standing\b|\bhydrant\b", t):
        return "illegal parking"
    if re.search(r"\bno heat\b|\bhot water\b|\bboiler\b|\bradiator\b", t):
        return "heat/hot water"
    if re.search(r"\bnoise\b|\bloud\b|\bmusic\b|\bbanging\b|\bparty\b", t):
        return "noise - residential"

    return ""


def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def fix_mojibake(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    if "Ã" in s or "â" in s:
        try:
            repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                return repaired
        except Exception:
            pass
    return s


def looks_spanish_or_non_english(text: str) -> bool:
    text = safe_str(text)
    if not text:
        return False
    if any(ord(ch) > 127 for ch in text):
        return True
    lower = text.lower()
    return any(marker in lower for marker in SPANISH_MARKERS)


def ensure_translation_dependencies() -> None:
    global _torch, _tqdm, _MarianMTModel, _MarianTokenizer, _DEVICE
    if _torch is not None:
        return
    import torch
    from tqdm.auto import tqdm
    from transformers import MarianMTModel, MarianTokenizer
    _torch = torch
    _tqdm = tqdm
    _MarianMTModel = MarianMTModel
    _MarianTokenizer = MarianTokenizer
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_translation_model(lang: str):
    ensure_translation_dependencies()
    if lang not in _TRANSLATION_MODEL_CACHE:
        model_name = MARIAN_MODEL_MAP[lang]
        LOGGER.info("Loading translation model for %s: %s", lang, model_name)
        tokenizer = _MarianTokenizer.from_pretrained(model_name)
        model = _MarianMTModel.from_pretrained(model_name).to(_DEVICE)
        model.eval()
        _TRANSLATION_MODEL_CACHE[lang] = (tokenizer, model)
    return _TRANSLATION_MODEL_CACHE[lang]


def translate_unique_values(values: Iterable[str], lang: str, batch_size: int = 32, max_length: int = 128) -> Dict[str, str]:
    vals = [safe_str(v) for v in values if safe_str(v)]
    vals = list(dict.fromkeys(vals))
    if not vals:
        return {}
    tokenizer, model = get_translation_model(lang)
    translation_map: Dict[str, str] = {}
    LOGGER.info("Translating %s unique values for language=%s", len(vals), lang)
    for i in _tqdm(range(0, len(vals), batch_size), desc=f"Translating {lang}"):
        batch = vals[i:i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(_DEVICE) for k, v in encoded.items()}
        with _torch.no_grad():
            generated = model.generate(**encoded)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for src, tgt in zip(batch, decoded):
            translation_map[src] = tgt
    return translation_map


def load_artifacts():
    required = [CATEGORY_MODEL_FILE, CATEGORY_LABEL_FILE, ROUTING_MODEL_FILE, ROUTING_LABEL_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))
    LOGGER.info("Loading category pipeline: %s", CATEGORY_MODEL_FILE)
    category_model = joblib.load(CATEGORY_MODEL_FILE)
    category_label_encoder = joblib.load(CATEGORY_LABEL_FILE)
    LOGGER.info("Loading routing pipeline: %s", ROUTING_MODEL_FILE)
    routing_model = joblib.load(ROUTING_MODEL_FILE)
    routing_label_encoder = joblib.load(ROUTING_LABEL_FILE)
    return {
        "category_model": category_model,
        "category_label_encoder": category_label_encoder,
        "routing_model": routing_model,
        "routing_label_encoder": routing_label_encoder,
    }


def load_agency_name_map() -> dict:
    """
    Return a static mapping of agency code -> agency name.
    This replaces file-based lookup for reliability and speed.
    """
    agency_map = {
        "DCWP": "Department of Consumer and Worker Protection",
        "DEP": "Department of Environmental Protection",
        "DHS": "Department of Homeless Services",
        "DOB": "Department of Buildings",
        "DOE": "Department of Education",
        "DOHMH": "Department of Health and Mental Hygiene",
        "DOT": "Department of Transportation",
        "DPR": "Department of Parks and Recreation",
        "DSNY": "Department of Sanitation",
        "HPD": "Department of Housing Preservation and Development",
        "NYPD": "New York City Police Department",
        "OOS": "Office of the Sheriff",
        "OTI": "Office of Technology and Innovation",
        "TLC": "Taxi and Limousine Commission",
    }

    LOGGER.info("Loaded static agency_name mapping with %s entries", len(agency_map))
    return agency_map


def detect_test_file() -> Path:
    LOGGER.info("Current working dir: %s", Path.cwd())
    LOGGER.info("Resolved TEST_DATA_DIR: %s", TEST_DATA_DIR)

    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"test_data folder not found: {TEST_DATA_DIR}")

    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name != OUTPUT_FILE.name and not p.name.startswith(".")
    ]

    LOGGER.info("CSV candidates found: %s", [p.name for p in candidates])

    if not candidates:
        raise FileNotFoundError(f"No suitable test CSV found in: {TEST_DATA_DIR}")

    preferred_names = [
        "urbanpulse_311_complaints_bilingual_test.csv",
        "test.csv",
        "test_data.csv",
    ]

    for name in preferred_names:
        for path in candidates:
            if path.name.lower() == name.lower():
                return path

    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)[0]



def translate_spanish_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = ["complaint_type", "descriptor", "resolution_description"]
    LOGGER.info("Preparing text columns")
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").map(fix_mojibake)
        df[f"{col}_original"] = df[col]
    df["lang_detection_text"] = (
        df["complaint_type"].astype(str)
        + " | "
        + df["descriptor"].astype(str)
        + " | "
        + df["resolution_description"].astype(str)
    )
    LOGGER.info("Flagging likely Spanish rows")
    df["needs_translation"] = df["lang_detection_text"].apply(looks_spanish_or_non_english)
    candidate_idx = df.index[df["needs_translation"]]
    LOGGER.info("Rows flagged for translation: %s / %s", len(candidate_idx), len(df))
    if len(candidate_idx) == 0:
        return df
    for col in text_cols:
        unique_vals = df.loc[candidate_idx, col].dropna().astype(str).str.strip()
        unique_vals = unique_vals[unique_vals != ""].unique().tolist()
        if unique_vals:
            trans_map = translate_unique_values(
                unique_vals,
                "es",
                16 if col == "resolution_description" else 32,
                256 if col == "resolution_description" else 64,
            )
            df.loc[candidate_idx, col] = df.loc[candidate_idx, col].apply(
                lambda x: trans_map.get(str(x).strip(), x)
                if pd.notna(x) and str(x).strip() != ""
                else x
            )
    return df


def build_category_text(df: pd.DataFrame) -> pd.Series:
    complaint_type = df.get("complaint_type", pd.Series([""] * len(df), index=df.index)).fillna("").map(normalize_complaint_type)
    descriptor = df.get("descriptor", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    resolution = df.get("resolution_description", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)

    # If complaint_type is blank, infer a complaint_type-like prefix from free text.
    inferred = (descriptor + " " + resolution).map(infer_complaint_type_from_free_text)
    complaint_type = complaint_type.where(complaint_type != "", inferred)

    return (complaint_type + " | " + descriptor + " | " + resolution).str.strip(" |")


def build_routing_text(df: pd.DataFrame) -> pd.Series:
    complaint_type = df.get("complaint_type", pd.Series([""] * len(df), index=df.index)).fillna("").map(normalize_complaint_type)
    descriptor = df.get("descriptor", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    resolution = df.get("resolution_description", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)

    # Match training format. If complaint_type is absent, infer it from the free text.
    inferred = (descriptor + " " + resolution).map(infer_complaint_type_from_free_text)
    complaint_type = complaint_type.where(complaint_type != "", inferred)

    return (complaint_type + " | " + descriptor + " | " + resolution).str.strip(" |")


def compute_confidence_scores(model, X_text: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_text)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba.astype(float)
        return proba.max(axis=1).astype(float)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_text)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.maximum(probs, 1.0 - probs).astype(float)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs.max(axis=1).astype(float)
    return np.ones(len(X_text), dtype=float)


def predict_dual(artifacts, test_data: pd.DataFrame, agency_map: dict) -> pd.DataFrame:
    category_text = build_category_text(test_data)
    routing_text = build_routing_text(test_data)

    category_model = artifacts["category_model"]
    category_le = artifacts["category_label_encoder"]
    routing_model = artifacts["routing_model"]
    routing_le = artifacts["routing_label_encoder"]

    LOGGER.info("Predicting complaint category using pipeline")
    cat_pred_encoded = category_model.predict(category_text)
    cat_pred_labels = category_le.inverse_transform(np.asarray(cat_pred_encoded).astype(int))
    cat_conf = compute_confidence_scores(category_model, category_text)

    LOGGER.info("Predicting routing agency using pipeline")
    route_pred_encoded = routing_model.predict(routing_text)
    route_pred_labels = routing_le.inverse_transform(np.asarray(route_pred_encoded).astype(int))
    route_conf = compute_confidence_scores(routing_model, routing_text)

    return pd.DataFrame({
        "id": test_data["id"],
        "predicted_category": cat_pred_labels,
        "category_confidence": np.round(cat_conf, 6),
        "recommended_agency": route_pred_labels,
        "recommended_agency_name": pd.Series(route_pred_labels).map(agency_map).fillna("Unknown Agency"),
        "routing_confidence": np.round(route_conf, 6),
    })


def main():
    LOGGER.info("Dual predict.py started")
    artifacts = load_artifacts()
    agency_map = load_agency_name_map()

    print("cwd:", Path.cwd())
    print("script:", Path(__file__).resolve())
    print("test_data exists:", TEST_DATA_DIR.exists())
    print("files:", list(TEST_DATA_DIR.glob("*")) if TEST_DATA_DIR.exists() else "missing")


    test_file = detect_test_file()
    LOGGER.info("Detected test file: %s", test_file)
    test_df = pd.read_csv(test_file)
    LOGGER.info("Loaded test data | shape=%s", test_df.shape)
    if "id" not in test_df.columns:
        test_df = test_df.copy()
        test_df.insert(0, "id", np.arange(1, len(test_df) + 1))
    processed = translate_spanish_candidate_rows(test_df)
    results = predict_dual(artifacts, processed, agency_map)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    LOGGER.info("Predictions saved to %s", OUTPUT_FILE)
    LOGGER.info("Output columns: %s", list(results.columns))
    LOGGER.info("Dual prediction complete")


if __name__ == "__main__":
    main()
