"""SentenceTransformer embeddings (no LangChain wrapper)."""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(name, device="cpu")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode([text], normalize_embeddings=True)[0],
            dtype=np.float32,
        )
