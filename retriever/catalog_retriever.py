"""Retriever agent: semantic search over catalog FAISS index."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import TOP_K_RETRIEVAL
from vectorstore.faiss_store import FAISSCatalogStore


@dataclass
class RetrievedChunk:
    """UI-friendly chunk with citation fields."""

    text: str
    source: str
    page: int
    chunk_id: str
    score: float | None = None


class CatalogRetrieverAgent:
    """Wraps FAISS retrieval; used by planner and verifier."""

    def __init__(self, store: FAISSCatalogStore, k: int | None = None) -> None:
        self.store = store
        self.k = k if k is not None else TOP_K_RETRIEVAL

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        pairs = self.store.similarity_search_with_scores(query, k=self.k)
        out: List[RetrievedChunk] = []
        for doc, score in pairs:
            meta = doc.metadata or {}
            out.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 0) or 0),
                    chunk_id=str(meta.get("chunk_id", "")),
                    score=float(score) if score is not None else None,
                )
            )
        return out
