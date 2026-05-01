import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path



import re
from typing import Dict, Iterable, Tuple

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



from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

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















# ===========================================================================
# 1. PAGE CONFIGURATION & BRANDING
# ===========================================================================
st.set_page_config(
    page_title="UrbanPulse Analytics | Nova Haven",
    page_icon="🏙️",
    layout="wide",
)

#st.title("AI Capstone Dashboard")
#st.write("Select a model from the sidebar to make predictions.")

# Sidebar Branding
st.sidebar.title("🏙️ UrbanPulse")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Select Intelligence Module",
    [
        "Home", 
        "Model 1: Traffic Severity (ML)", 
        "Model 2: Resource Allocation (DNN)", 
        "Model 3: Road Inspection (CNN)", 
        "Model 4: 311 Classifier (NLP)", 
        "Model 5: Innovation Module"
    ]
)

# ===========================================================================
# 2. CACHED MODEL LOADERS
# ===========================================================================
@st.cache_resource
def load_ml_model():
    import joblib
    return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")

@st.cache_resource
def load_dnn_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")

@st.cache_resource
def load_cnn_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")




### functions for model 4

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    return agency_map



def translate_spanish_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = ["complaint_type", "descriptor", "resolution_description"]

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

    df["needs_translation"] = df["lang_detection_text"].apply(looks_spanish_or_non_english)
    candidate_idx = df.index[df["needs_translation"]]

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
    return (
        df.get("descriptor", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
        + " "
        + df.get("resolution_description", pd.Series([""] * len(df), index=df.index)).fillna("").map(clean_text)
    ).str.strip()