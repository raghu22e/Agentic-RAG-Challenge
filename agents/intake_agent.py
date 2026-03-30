"""Intake agent: structured profile + clarifying questions."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from prompts.templates import INTAKE_SYSTEM
from src.config import RAG_SKIP_INTAKE_LLM
from src.llm_provider import get_chat_model


def _parse_json_blob(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


class IntakeAgent:
    def __init__(self) -> None:
        self.llm = get_chat_model()

    def update_profile(
        self,
        user_message: str,
        prior_profile: Dict[str, Any] | None,
        recent_turns: List[Tuple[str, str]] | None = None,
    ) -> Dict[str, Any]:
        prior = dict(prior_profile or {})
        if RAG_SKIP_INTAKE_LLM:
            prior.setdefault("clarifying_questions", [])
            return prior
        hist = ""
        if recent_turns:
            for u, a in recent_turns[-6:]:
                hist += f"User: {u}\nAssistant: {a}\n"
        human = (
            f"Current profile JSON (may be empty): {json.dumps(prior)}\n\n"
            f"Recent dialogue:\n{hist}\n"
            f"Latest user message: {user_message}\n\n"
            "Merge/update the profile. Return JSON only."
        )
        msg = self.llm.invoke(
            [SystemMessage(content=INTAKE_SYSTEM), HumanMessage(content=human)]
        )
        parsed = _parse_json_blob(getattr(msg, "content", str(msg)))
        merged = {**prior, **{k: v for k, v in parsed.items() if v is not None}}
        for key in ("program", "interests_electives"):
            if key in parsed and parsed[key]:
                merged[key] = parsed[key]
        if "completed_courses" in parsed and isinstance(parsed["completed_courses"], list):
            merged["completed_courses"] = list(
                dict.fromkeys((prior.get("completed_courses") or []) + parsed["completed_courses"])
            )
        if "semester" in parsed:
            merged["semester"] = parsed["semester"]
        merged["clarifying_questions"] = parsed.get("clarifying_questions") or []
        return merged
