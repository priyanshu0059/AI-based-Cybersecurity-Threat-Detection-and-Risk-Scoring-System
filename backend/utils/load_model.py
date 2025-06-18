import joblib
import os
import logging
import json
from pathlib import Path

# === Setup Logger ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # root dir of project
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'full_pipeline.pkl'  # 🔁 Changed to full_pipeline
METADATA_PATH = MODEL_DIR / 'model_metadata.json'

# === Cache variables ===
_model = None
_metadata = None

# === Custom Error ===
class ModelLoadError(Exception):
    """Raised when model or metadata fails to load."""
    pass

# === Load Model Pipeline ===
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelLoadError(f"❌ Model not found at: {MODEL_PATH}")
        logger.info(f"✅ Loading full model pipeline from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

# === Load Metadata (optional) ===
def get_metadata():
    global _metadata
    if _metadata is None:
        if not METADATA_PATH.exists():
            logger.warning(f"⚠️ Metadata file not found at: {METADATA_PATH}")
            return {}
        with open(METADATA_PATH, 'r') as f:
            _metadata = json.load(f)
        logger.info(f"📄 Metadata loaded from: {METADATA_PATH}")
    return _metadata
