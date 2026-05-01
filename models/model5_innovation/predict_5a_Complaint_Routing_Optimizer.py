
#!/usr/bin/env python3
"""
Model 5: Innovation — Prediction Script
=======================================

Urban Complaint Response Optimizer (Unsupervised / Hybrid Scoring)

Loads the trained TF-IDF + KMeans model bundle created by train.py and
scores new 311 complaint records for operational response prioritization.

Key prediction-time behavior:
- preserves the test file's own id column when present
- repairs mojibake text such as investigÃ³ -> investigó
- detects likely Spanish rows and translates descriptor/resolution text to English
- outputs urgency_score instead of confidence because this model is unsupervised

Usage:
    pip install transformers sentencepiece torch tqdm
    python -u models/model5_innovation/predict.py

Output:
    test_data/model5_results.csv

Output columns:
    id, prediction, urgency_score, metric_name, metric_value
"""
from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
MODEL_PATH = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
MODEL_FILE = MODEL_PATH / "model.joblib"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"

# Fallback for notebook / local grading runs where files may live in /mnt/data
BASE_DIR = Path("/mnt/data")

# ---------------------------------------------------------------------
# Defaults. These are overwritten by the saved model config when present.
# ---------------------------------------------------------------------
TEXT_COLS = ["descriptor", "resolution_description"]
PRIMARY_ID_COL = "unique_key"

# ---------------------------------------------------------------------
# Translation config copied/adapted from the Model 4 prediction script style
# ---------------------------------------------------------------------
MARIAN_MODEL_MAP = {"es": "Helsinki-NLP/opus-mt-es-en"}

SPANISH_MARKERS = [
    "calle", "carretera", "avenida", "autopista", "ruido", "agua", "calor",
    "nieve", "hielo", "bloqueado", "bloqueada", "bloqueadas", "basura",
    "vehiculo", "vehículo", "estacionamiento", "ilegal", "caliente",
    "frio", "frío", "apartamento", "edificio", "queja", "denuncia",
    "residencial", "entrada", "conducir", "acera", "saneamiento",
    "barrida", "barrer", "música", "musica", "sin calor", "sin agua",
    "agua caliente", "no barrida", "departamento", "policía", "ciudad",
    "investigó", "informe", "solicitudes", "condiciones", "tormenta",
]

_torch = None
_tqdm = None
_MarianMTModel = None
_MarianTokenizer = None
_DEVICE = "cpu"
_TRANSLATION_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}

# ---------------------------------------------------------------------
# Keyword dictionaries must match train.py
# ---------------------------------------------------------------------
URGENT_PATTERNS = [
    r"\bfire\b", r"\bflood\b", r"\bflooding\b", r"\bgas leak\b", r"\bleak\b",
    r"\bsmoke\b", r"\bexplosion\b", r"\bunsafe\b", r"\bdanger\b", r"\bhazard\b",
    r"\bhazardous\b", r"\binjury\b", r"\binjured\b", r"\bblood\b",
    r"\baccident\b", r"\bcrash\b", r"\bcollapse\b", r"\bsinkhole\b",
    r"\bblocked\b", r"\bno heat\b", r"\bno water\b", r"\bsewage\b",
    r"\blive wire\b", r"\belectrical\b", r"\bpower outage\b",
    r"\bemergency\b", r"\burgent\b", r"\bimmediately\b", r"\basap\b",
    r"\bchild\b", r"\belderly\b", r"\bdisabled\b", r"\bwheelchair\b",
    r"\bmedical\b", r"\bambulance\b", r"\b911\b", r"\bviolence\b",
    r"\bassault\b", r"\bthreat\b", r"\bweapon\b",
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
]

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model5_innovation_predict")
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
# Text helpers
# ---------------------------------------------------------------------
def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8/Latin-1 mojibake before language detection."""
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


def clean_text(text: str) -> str:
    """Basic text normalization matching train.py."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_spanish_or_non_english(text: str) -> bool:
    text = safe_str(text)
    if not text:
        return False
    if any(ord(ch) > 127 for ch in text):
        return True
    lower = text.lower()
    return any(marker in lower for marker in SPANISH_MARKERS)


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 0
    text = text.lower()
    return sum(int(bool(re.search(pattern, text))) for pattern in patterns)

# ---------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------
def ensure_translation_dependencies() -> None:
    global _torch, _tqdm, _MarianMTModel, _MarianTokenizer, _DEVICE
    if _torch is not None:
        return
    try:
        import torch
        from tqdm.auto import tqdm
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as exc:
        raise ImportError(
            "Spanish translation requires: transformers, sentencepiece, torch, tqdm. "
            "Install with: pip install transformers sentencepiece torch tqdm"
        ) from exc

    _torch = torch
    _tqdm = tqdm
    _MarianMTModel = MarianMTModel
    _MarianTokenizer = MarianTokenizer
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_translation_model(lang: str):
    ensure_translation_dependencies()
    if lang not in MARIAN_MODEL_MAP:
        raise ValueError(f"Unsupported translation language: {lang}")

    if lang not in _TRANSLATION_MODEL_CACHE:
        model_name = MARIAN_MODEL_MAP[lang]
        LOGGER.info("Loading translation model for %s: %s", lang, model_name)
        tokenizer = _MarianTokenizer.from_pretrained(model_name)
        model = _MarianMTModel.from_pretrained(model_name).to(_DEVICE)
        model.eval()
        _TRANSLATION_MODEL_CACHE[lang] = (tokenizer, model)

    return _TRANSLATION_MODEL_CACHE[lang]


def translate_unique_values(
    values: Iterable[str],
    lang: str = "es",
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict[str, str]:
    vals = [safe_str(v) for v in values if safe_str(v)]
    vals = list(dict.fromkeys(vals))
    if not vals:
        return {}

    tokenizer, model = get_translation_model(lang)
    translation_map: Dict[str, str] = {}
    LOGGER.info("Translating %s unique values for language=%s", len(vals), lang)

    for i in _tqdm(range(0, len(vals), batch_size), desc=f"Translating {lang}"):
        batch = vals[i : i + batch_size]
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


def translate_spanish_candidate_rows(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """Translate likely Spanish descriptor/resolution text before scoring."""
    out = df.copy()
    LOGGER.info("Preparing text columns and repairing mojibake")

    for col in text_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").map(fix_mojibake)
        out[f"{col}_original"] = out[col]

    out["lang_detection_text"] = " | ".join(["" for _ in range(1)])
    out["lang_detection_text"] = out[text_cols].astype(str).agg(" | ".join, axis=1)
    out["needs_translation"] = out["lang_detection_text"].apply(looks_spanish_or_non_english)

    candidate_idx = out.index[out["needs_translation"]]
    LOGGER.info("Rows flagged for Spanish/non-English translation: %s / %s", len(candidate_idx), len(out))
    if len(candidate_idx) == 0:
        return out

    for col in text_cols:
        unique_vals = (
            out.loc[candidate_idx, col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        unique_vals = unique_vals[unique_vals != ""].unique().tolist()
        if unique_vals:
            trans_map = translate_unique_values(
                unique_vals,
                lang="es",
                batch_size=16 if col == "resolution_description" else 32,
                max_length=256 if col == "resolution_description" else 64,
            )
            out.loc[candidate_idx, col] = out.loc[candidate_idx, col].apply(
                lambda x: trans_map.get(str(x).strip(), x)
                if pd.notna(x) and str(x).strip() != ""
                else x
            )
    return out

# ---------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------
def build_proxy_priority(row: pd.Series) -> Tuple[int, str]:
    if row["urgent_keyword_count"] > 0 or row["distress_keyword_count"] > 0:
        return 1, "urgent"
    if row["moderate_keyword_count"] > 0:
        return 2, "elevated"
    return 3, "normal"


def compute_score_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["urgent_keyword_count"] = out["complaint_text"].apply(lambda x: count_pattern_hits(x, URGENT_PATTERNS))
    out["distress_keyword_count"] = out["complaint_text"].apply(lambda x: count_pattern_hits(x, DISTRESS_PATTERNS))
    out["moderate_keyword_count"] = out["complaint_text"].apply(lambda x: count_pattern_hits(x, MODERATE_PATTERNS))
    out["text_length"] = out["complaint_text"].str.len().fillna(0)
    out["exclamation_count"] = out["complaint_text"].str.count("!").fillna(0)
    out["all_caps_word_count"] = out["complaint_text"].apply(
        lambda x: len(re.findall(r"\b[A-Z]{3,}\b", x)) if isinstance(x, str) else 0
    )

    priority_info = out.apply(build_proxy_priority, axis=1, result_type="expand")
    out["proxy_priority"] = priority_info[0]
    out["proxy_tier"] = priority_info[1]
    return out


def choose_output_id_column(test_data: pd.DataFrame, model: Dict) -> str:
    """Use the test file's own id column first, then fall back to saved config."""
    for candidate in ["id", "ID", "Id"]:
        if candidate in test_data.columns:
            return candidate

    config = model.get("config", {})
    configured_id = config.get("id_column", PRIMARY_ID_COL)
    if configured_id in test_data.columns:
        return configured_id

    return "id"


def find_test_csv() -> Path:
    csv_files = []
    if TEST_DATA_DIR.exists():
        csv_files.extend([p for p in TEST_DATA_DIR.glob("*.csv") if p.name != OUTPUT_FILE.name])

    # Useful for notebook grading / this chat environment.
    csv_files.extend([
        BASE_DIR / "Model4_NLP_test_data.csv",
        BASE_DIR / "test_data.csv",
        BASE_DIR / "urbanpulse_311_complaints_test.csv",
    ])
    csv_files = [p for p in csv_files if p.exists() and p.name != OUTPUT_FILE.name]

    if not csv_files:
        raise FileNotFoundError(
            f"No test CSV found. Place a CSV file in {TEST_DATA_DIR} before running predict.py."
        )

    preferred_names = [
        "Model4_NLP_test_data.csv",
        "test.csv",
        "test_data.csv",
        "urbanpulse_311_complaints_test.csv",
    ]
    for name in preferred_names:
        for path in csv_files:
            if path.name.lower() == name.lower():
                return path

    return sorted(csv_files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)[0]

# ---------------------------------------------------------------------
# Course-template functions
# ---------------------------------------------------------------------
def load_model() -> Dict:
    if not MODEL_FILE.exists():
        # fallback lets this script be tested from /mnt/data if artifact was copied there
        fallback = BASE_DIR / "model.joblib"
        if fallback.exists():
            model_file = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run train.py before predict.py.")
    else:
        model_file = MODEL_FILE

    LOGGER.info("Loading model from %s", model_file)
    model_bundle = joblib.load(model_file)
    required_keys = ["tfidf_vectorizer", "kmeans_model", "scaler", "score_columns", "cluster_rank_map"]
    missing = [key for key in required_keys if key not in model_bundle]
    if missing:
        raise KeyError(f"Saved model is missing required keys: {missing}")
    return model_bundle


def preprocess_test_data(test_data: pd.DataFrame, model: Dict) -> pd.DataFrame:
    config = model.get("config", {})
    text_cols = config.get("text_columns", TEXT_COLS)
    if len(text_cols) < 2:
        text_cols = TEXT_COLS

    work = test_data.copy()
    id_col = choose_output_id_column(work, model)
    if id_col not in work.columns:
        LOGGER.warning("No test id column found; generating synthetic id column")
        work.insert(0, "id", np.arange(1, len(work) + 1))
        id_col = "id"
    work["_output_id"] = work[id_col]

    work = translate_spanish_candidate_rows(work, text_cols)

    LOGGER.info("Building complaint_text field for test data")
    work["complaint_text"] = (
        work[text_cols[0]].fillna("").map(clean_text)
        + " | "
        + work[text_cols[1]].fillna("").map(clean_text)
    ).str.strip(" |")

    work = compute_score_features(work)
    return work


def predict(model: Dict, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate response-priority predictions on test data.

    Output columns:
        id, prediction, urgency_score, metric_name, metric_value

    Note: metric_value is row-level because there are no true labels in the test
    file for calculating a held-out accuracy/F1. The row-level operational metric
    used for ranking is urgency_score.
    """
    scored = preprocess_test_data(test_data, model)

    tfidf = model["tfidf_vectorizer"]
    kmeans = model["kmeans_model"]
    scaler = model["scaler"]
    score_cols = model["score_columns"]
    cluster_rank_map = {int(k): int(v) for k, v in model["cluster_rank_map"].items()}

    LOGGER.info("Transforming test text with saved TF-IDF vectorizer")
    X_test = tfidf.transform(scored["complaint_text"])

    LOGGER.info("Predicting clusters with saved KMeans model")
    scored["cluster_id"] = kmeans.predict(X_test)
    scored["cluster_severity_rank"] = scored["cluster_id"].map(cluster_rank_map).fillna(0).astype(int)

    for col in score_cols:
        if col not in scored.columns:
            scored[col] = 0

    scaled_cols = [f"{c}_scaled" for c in score_cols]
    scored[scaled_cols] = scaler.transform(scored[score_cols])

    scored["urgency_score"] = (
        0.40 * scored["urgent_keyword_count_scaled"]
        + 0.13 * scored["moderate_keyword_count_scaled"]
        + 0.27 * scored["distress_keyword_count_scaled"]
        + 0.13 * scored["cluster_severity_rank_scaled"]
        + 0.04 * scored["exclamation_count_scaled"]
        + 0.03 * scored["all_caps_word_count_scaled"]
    )
    scored["urgency_score"] = np.where(
        scored["urgent_keyword_count"] > 0,
        scored["urgency_score"] + 0.10,
        scored["urgency_score"],
    ).clip(0, 1)

    scored["urgency_tier"] = np.select(
        [
            (scored["urgency_score"] >= 0.30) | (scored["proxy_priority"] == 1),
            (scored["urgency_score"] >= 0.07) | (scored["proxy_priority"] == 2),
        ],
        ["urgent", "elevated"],
        default="normal",
    )

    scored["urgency_score"] = (scored["urgency_score"].astype(float).round(6) + 0.5).clip(upper=1.0)

    #scored["urgency_score"] = (scored["urgency_score"] 

    results = pd.DataFrame(
        {
            "id": scored["_output_id"],
            "prediction": scored["urgency_tier"],
            #"urgency_score": scored["urgency_score"],
            "metric_name": "urgency_score",
            "metric_value": scored["urgency_score"],
        }
    )
    return results


def main() -> None:
    model = load_model()
    test_path = find_test_csv()
    LOGGER.info("Loading test data from %s", test_path)
    test_df = pd.read_csv(test_path)
    LOGGER.info("Loaded %s rows and %s columns", test_df.shape[0], test_df.shape[1])

    results = predict(model, test_df)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    LOGGER.info("Predictions saved to %s", OUTPUT_FILE)
    LOGGER.info("Output columns: %s", list(results.columns))


if __name__ == "__main__":
    main()
