import os
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# ── Project Root ────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Paths (Overrideable via ENV) ──────────────────────────────────────────
# We use the 'PP_' prefix to align with the Text-Only version
DATASET_DIR = os.getenv("PP_DATA_DIR", os.path.join(BASE_DIR, "data", "extracted_papers"))
TREES_DIR   = os.getenv("PP_TREES_DIR", os.path.join(BASE_DIR, "data", "trees"))
INDEX_DIR   = os.getenv("PP_INDEX_DIR", os.path.join(BASE_DIR, "data", "index"))
RESULTS_DIR = os.getenv("PP_RESULTS_DIR", os.path.join(BASE_DIR, "results"))

# ── Model Config ────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
NOISE_FILTER_MODEL = "gemini-3.1-flash-lite-preview"
SYNTH_MODEL        = "gemini-3.1-flash-lite-preview"
VISION_FILTER      = False # Set to True for high-fidelity image verification (adds ~30s latency)
