"""End-to-end agent pipeline: intake -> retrieve -> plan -> verify."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from agents.intake_agent import IntakeAgent
from agents.parse_response import ParsedSections, parse_sections
from agents.planner_agent import PlannerAgent
from agents.verifier_agent import VerifierAgent, _join_sections
from retriever.catalog_retriever import CatalogRetrieverAgent, RetrievedChunk
from src.config import RAG_FAST


def _fallback_parsed_from_chunks(
    chunks: List[RetrievedChunk], user_message: str
) -> ParsedSections:
    """Last resort: show readable excerpts when the model still returns no Chat reply."""
    lines: List[str] = []
    cites: List[str] = []
    for c in chunks[:5]:
        snippet = (c.text or "").strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        lines.append(f"• {snippet}")
        cites.append(f"- {c.source} (page {c.page})")
    q = user_message.strip()
    if len(q) > 160:
        q = q[:160] + "…"
    body = (
        "Here are the most relevant passages from your university catalog (indexed documents) for "
        f"your question. Read them together with the **Sources** list on the right.\n\n"
        + "\n".join(lines)
    )
    return ParsedSections(
        answer_or_plan=body,
        why="",
        citations="\n".join(cites),
        clarifying_questions="None",
        assumptions="",
        raw="",
    )


@dataclass
class OrchestratorResult:
    reply_text: str
    parsed: Any
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)


class CoursePlanningOrchestrator:
    def __init__(self, retriever: CatalogRetrieverAgent) -> None:
        self.retriever = retriever
        self.intake = IntakeAgent()
        self.planner = PlannerAgent()
        # Fast mode: rule-based verify only (saves one LLM round-trip).
        self.verifier = VerifierAgent(use_llm_revision=not RAG_FAST)

    def run_turn(
        self,
        user_message: str,
        profile: Dict[str, Any] | None,
        dialogue: List[Tuple[str, str]] | None = None,
    ) -> OrchestratorResult:
        steps: List[str] = []
        steps.append("Intake Agent: updating student profile from message.")
        prof = self.intake.update_profile(user_message, profile, dialogue)

        rq_parts = [user_message]
        if prof.get("program"):
            rq_parts.append(str(prof["program"]))
        if prof.get("completed_courses"):
            rq_parts.append("completed: " + ",".join(prof["completed_courses"]))
        retrieval_query = " \n".join(rq_parts)
        steps.append("Catalog Retriever Agent: embedding search over FAISS index.")
        chunks = self.retriever.retrieve(retrieval_query)
        steps.append(f"Retrieved {len(chunks)} chunks.")

        steps.append("Planner Agent: generating conversational reply with citations.")
        draft = self.planner.run(user_message, prof, chunks)

        steps.append("Verifier Agent: checking support in retrieved excerpts.")
        final_text, vsteps = self.verifier.verify(draft, chunks)
        steps.extend(vsteps)

        parsed = parse_sections(final_text)
        if not (parsed.answer_or_plan or "").strip() and chunks:
            steps.append("Planner Agent: repair pass (empty Chat reply with retrieved evidence).")
            draft2 = self.planner.run_repair(user_message, prof, chunks)
            final_text, vsteps2 = self.verifier.verify(draft2, chunks)
            steps.extend(vsteps2)
            parsed = parse_sections(final_text)

        if not (parsed.answer_or_plan or "").strip() and chunks:
            steps.append("Fallback: excerpt-based reply (model still returned no usable Chat reply).")
            parsed = _fallback_parsed_from_chunks(chunks, user_message)
            final_text = _join_sections(parsed)

        return OrchestratorResult(
            reply_text=final_text,
            parsed=parsed,
            retrieved_chunks=chunks,
            reasoning_steps=steps,
            profile=prof,
        )
