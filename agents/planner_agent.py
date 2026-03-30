"""Planner agent: RAG-conditioned course planning."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from prompts.templates import PLANNER_REPAIR_SYSTEM, PLANNER_SYSTEM
from retriever.catalog_retriever import RetrievedChunk
from src.llm_provider import get_chat_model


def _profile_for_planner(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Intake may attach clarifying_questions; never pass those to the planner—they skew answers."""
    out: Dict[str, Any] = {}
    for key in ("program", "semester", "completed_courses", "interests_electives"):
        if key not in profile:
            continue
        v = profile.get(key)
        if v is None or v == "" or v == []:
            continue
        out[key] = v
    return out


def _format_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"--- Excerpt {i} ---\n"
            f"Source: {c.source} | Page: {c.page} | chunk_id: {c.chunk_id}\n"
            f"{c.text}\n"
        )
    return "\n".join(blocks)


class PlannerAgent:
    def __init__(self) -> None:
        self.llm = get_chat_model()

    def run(
        self,
        user_message: str,
        profile: Dict[str, Any],
        chunks: List[RetrievedChunk],
    ) -> str:
        ctx = _format_context(chunks)
        prof = json.dumps(_profile_for_planner(profile), ensure_ascii=False)
        human = (
            f"Student profile (JSON): {prof}\n\n"
            f"Catalog excerpts (only evidence):\n{ctx}\n\n"
            f"User request:\n{user_message}\n\n"
            "Use the Chat reply, Sources, and Clarifying Questions sections exactly as specified."
        )
        msg = self.llm.invoke(
            [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=human)]
        )
        return str(getattr(msg, "content", msg))

    def run_repair(
        self,
        user_message: str,
        profile: Dict[str, Any],
        chunks: List[RetrievedChunk],
    ) -> str:
        """Second pass when Chat reply was empty but chunks exist."""
        ctx = _format_context(chunks)
        prof = json.dumps(_profile_for_planner(profile), ensure_ascii=False)
        human = (
            f"Student profile (JSON): {prof}\n\n"
            f"Catalog excerpts (only evidence):\n{ctx}\n\n"
            f"User request:\n{user_message}\n\n"
            "The previous model output had an empty or missing Chat reply. Regenerate with a full Chat reply."
        )
        msg = self.llm.invoke(
            [SystemMessage(content=PLANNER_REPAIR_SYSTEM), HumanMessage(content=human)]
        )
        return str(getattr(msg, "content", msg))
