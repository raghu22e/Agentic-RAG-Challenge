"""Lightweight document chunk (replaces LangChain Document for local RAG)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DocChunk:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
