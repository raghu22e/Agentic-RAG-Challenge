"""Paths and model configuration."""
import os
from pathlib import Path


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All catalog PDFs live under data/ in themed subfolders (see data/README.md).
DATA_ROOT = PROJECT_ROOT / "data"

# Legacy single-folder layout (still scanned if present)
LEGACY_PDF_DIR = DATA_ROOT / "pdfs"

VECTOR_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Backwards compatibility: code that still imports DATA_DIR uses courses bucket for uploads
DATA_DIR = DATA_ROOT / "courses"

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Speed: RAG_FAST=1 uses fewer chunks and skips verifier LLM revision (rule-only verify).
RAG_FAST = _env_bool("RAG_FAST")
RAG_SKIP_INTAKE_LLM = _env_bool("RAG_SKIP_INTAKE_LLM")
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K", "5" if RAG_FAST else "8"))

# LLM HTTP timeout (seconds)
LLM_TIMEOUT_SEC = float(os.environ.get("LLM_TIMEOUT_SEC", "90"))

# Slightly above 0 for natural chatbot phrasing; set LLM_TEMPERATURE=0 for maximum determinism.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.25"))

# Sarvam AI (OpenAI-compatible chat API) — https://docs.sarvam.ai/api-reference-docs/getting-started/quickstart
SARVAM_API_BASE = os.environ.get("SARVAM_API_BASE", "https://api.sarvam.ai/v1")
SARVAM_CHAT_MODEL = os.environ.get("SARVAM_CHAT_MODEL", "sarvam-30b")

# OpenAI (optional alternative if SARVAM_API_KEY is not set)
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
