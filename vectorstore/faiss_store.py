"""FAISS index persistence (IndexFlatIP + normalized embeddings)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss

from embeddings.embedder import Embedder
from src.documents import DocChunk


class FAISSCatalogStore:
    def __init__(self, index_dir: Path, embedder: Embedder) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[Dict[str, Any]] = []

    def build(self, documents: List[DocChunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts = [d.page_content for d in documents]
        self._chunks = []
        for d in documents:
            self._chunks.append({"text": d.page_content, "metadata": dict(d.metadata)})
        embs = self.embedder.embed_documents(texts)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        self._index = index
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> faiss.IndexFlatIP:
        if self._index is not None and self._chunks:
            return self._index
        idx_path = self.index_dir / "index.faiss"
        ck_path = self.index_dir / "chunks.pkl"
        if not idx_path.exists() or not ck_path.exists():
            raise FileNotFoundError(f"FAISS index missing at {self.index_dir}. Run ingestion first.")
        self._index = faiss.read_index(str(idx_path))
        with open(ck_path, "rb") as f:
            self._chunks = pickle.load(f)
        return self._index

    def similarity_search_with_scores(self, query: str, k: int) -> List[Tuple[DocChunk, float]]:
        index = self.load()
        q = self.embedder.embed_query(query).reshape(1, -1)
        scores, idxs = index.search(q, min(k, len(self._chunks)))
        out: List[Tuple[DocChunk, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            row = self._chunks[i]
            dc = DocChunk(page_content=row["text"], metadata=row.get("metadata") or {})
            out.append((dc, float(score)))
        return out
