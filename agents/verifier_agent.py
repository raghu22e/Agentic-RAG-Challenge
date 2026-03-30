"""Verifier agent: citation / hallucination guardrails."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

from langchain_core.messages import HumanMessage, SystemMessage

from agents.parse_response import ParsedSections, parse_sections
from prompts.templates import VERIFIER_REVISION_SYSTEM
from retriever.catalog_retriever import RetrievedChunk
from src.llm_provider import DemoLLM, get_chat_model

_COURSE_CODE = re.compile(r"\b([A-Z]{2,6}\d{3,4}[A-Z]?)\b")


def _corpus(chunks: List[RetrievedChunk]) -> str:
    return "\n".join(c.text for c in chunks).upper()


def _sources(chunks: List[RetrievedChunk]) -> Set[str]:
    return {c.source for c in chunks}


def _cited_file_allowed(filename: str, allowed_sources: Set[str]) -> bool:
    """Match bare filename.pdf against full relative paths like courses/MA101.pdf."""
    fl = filename.lower()
    for s in allowed_sources:
        if s.lower() == fl or Path(s).name.lower() == fl:
            return True
    return False


def _codes_in(text: str) -> Set[str]:
    return {m.group(1).upper() for m in _COURSE_CODE.finditer(text or "")}


def rule_based_verify(parsed: ParsedSections, chunks: List[RetrievedChunk]) -> tuple[ParsedSections, List[str]]:
    """Return possibly modified sections + reasoning steps."""
    steps: List[str] = []
    corp = _corpus(chunks)
    allowed_sources = _sources(chunks)
    factual = " ".join([parsed.answer_or_plan, parsed.why, parsed.citations])
    codes = _codes_in(factual)
    unsupported = sorted(c for c in codes if c not in corp)
    if unsupported:
        steps.append(f"Flagged course codes not found in retrieved excerpts: {', '.join(unsupported)}")
        abstain = (
            "I do not have enough information in the retrieved catalog excerpts to verify "
            f"these course codes: {', '.join(unsupported)}. Please narrow the question or upload the relevant PDF."
        )
        parsed = ParsedSections(
            answer_or_plan=abstain,
            why="The verifier found course identifiers in the draft that do not appear in the retrieved chunks.",
            citations="; ".join(f"{c.source} p.{c.page}" for c in chunks[:5]),
            clarifying_questions=parsed.clarifying_questions
            or "Which program handbook should we search, and which semester are you planning for?",
            assumptions="Abstained rather than guess; only retrieved excerpts are trusted.",
            raw=parsed.raw,
        )
    # Citation filename sanity
    cite_text = parsed.citations or ""
    mentioned_files = re.findall(r"([\w\-]+\.pdf)", cite_text, re.IGNORECASE)
    bad = [f for f in mentioned_files if not _cited_file_allowed(f, allowed_sources)]
    if bad:
        steps.append(f"Citations mention files not in retrieval window: {bad}")
    if not chunks:
        steps.append("No chunks retrieved; forcing abstention.")
        parsed = ParsedSections(
            answer_or_plan="I do not have enough information in the document index to answer this.",
            why="Retriever returned no excerpts.",
            citations="",
            clarifying_questions="Could you rephrase using a course code or program name from your catalog PDFs?",
            assumptions="Empty retrieval.",
            raw=parsed.raw,
        )
    return parsed, steps


class VerifierAgent:
    def __init__(self, use_llm_revision: bool = True) -> None:
        self.llm = get_chat_model()
        self.use_llm_revision = use_llm_revision

    def verify(self, draft_text: str, chunks: List[RetrievedChunk]) -> tuple[str, List[str]]:
        parsed = parse_sections(draft_text)
        parsed, steps = rule_based_verify(parsed, chunks)
        joined = _join_sections(parsed)
        if not self.use_llm_revision:
            return joined, steps
        import os

        if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
            return joined, steps + ["Skipped LLM verifier revision in DEMO_MODE."]

        if isinstance(self.llm, DemoLLM):
            return joined, steps + ["Skipped verifier LLM (demo / missing API key — rule checks only)."]

        # Do not re-expand claims after a hard abstention from rule-based checks
        if "do not have enough information" in (parsed.answer_or_plan or "").lower():
            return joined, steps + ["Skipped LLM verifier revision after rule-based abstention."]

        ctx = "\n\n".join(f"[{c.source} p.{c.page}]\n{c.text}" for c in chunks)
        human = (
            f"DRAFT:\n{draft_text}\n\nEXCERPTS:\n{ctx}\n\n"
            "Return the full corrected response with the same section headers as in the draft "
            "(Chat reply, Sources, Clarifying Questions; keep optional Why/Assumptions only if present)."
        )
        msg = self.llm.invoke(
            [
                SystemMessage(content=VERIFIER_REVISION_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        revised = str(getattr(msg, "content", msg))
        return revised, steps + ["LLM verifier revision applied."]


def _join_sections(p: ParsedSections) -> str:
    """Serialize for storage/API; chat-first headers, optional Why/Assumptions for verifier abstentions."""
    blocks: List[str] = [
        "Chat reply:\n" + (p.answer_or_plan or "").strip(),
        "Sources:\n" + (p.citations or "").strip(),
        "Clarifying Questions:\n" + (p.clarifying_questions or "").strip(),
    ]
    if (p.why or "").strip():
        blocks.append("Why:\n" + (p.why or "").strip())
    if (p.assumptions or "").strip():
        blocks.append("Assumptions:\n" + (p.assumptions or "").strip())
    return "\n\n".join(blocks)
