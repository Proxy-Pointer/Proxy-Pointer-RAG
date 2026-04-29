"""
Proxy-Pointer: Centralized Configuration

All paths are relative to the project root. Override via .env or environment variables.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# ── Project Root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ── Environment ─────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Paths ───────────────────────────────────────────────────────────────
PDF_DIR       = Path(os.getenv("PP_PDF_DIR",       PROJECT_ROOT / "data" / "pdf"))
DATA_DIR      = Path(os.getenv("PP_DATA_DIR",      PROJECT_ROOT / "data" / "documents"))
TREES_DIR     = Path(os.getenv("PP_TREES_DIR",     PROJECT_ROOT / "data" / "trees"))
INDEX_DIR     = Path(os.getenv("PP_INDEX_DIR",      PROJECT_ROOT / "data" / "index"))
RESULTS_DIR   = Path(os.getenv("PP_RESULTS_DIR",    PROJECT_ROOT / "data" / "results"))

# ── LlamaParse ──────────────────────────────────────────────────────────
# Options: "cost_effective" (v2 default), "agentic", or "agentic_plus" (best for complex docs)
LLAMA_PARSE_TIER = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

# ── Models ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
NOISE_FILTER_MODEL = "gemini-3.1-flash-lite-preview"
SYNTH_MODEL        = "gemini-3.1-flash-lite-preview"

