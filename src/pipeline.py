"""Build or rebuild FAISS index from data/pdfs."""
from __future__ import annotations

from pathlib import Path

from embeddings.embedder import Embedder
from src.config import DATA_DIR, VECTOR_DIR
from src.ingestion import ingest_all
from vectorstore.faiss_store import FAISSCatalogStore


def build_faiss_index(pdf_dir: Path | None = None, vector_dir: Path | None = None) -> int:
    pdf_dir = pdf_dir or DATA_DIR
    vector_dir = vector_dir or VECTOR_DIR
    docs = ingest_all(pdf_dir)
    if not docs:
        raise RuntimeError(f"No PDFs found under {pdf_dir}. Run data/generate_synthetic_catalogs.py")
    embedder = Embedder()
    store = FAISSCatalogStore(vector_dir, embedder)
    store.build(docs)
    return len(docs)
