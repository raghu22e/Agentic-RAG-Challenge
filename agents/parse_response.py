"""Parse structured section blocks from model output."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class ParsedSections:
    answer_or_plan: str
    why: str
    citations: str
    clarifying_questions: str
    assumptions: str
    raw: str


_HEADERS = [
    # Chat-first layout (preferred) and legacy Answer / Plan
    ("answer", r"(?:Chat reply|Reply|Answer\s*/\s*Plan)\s*:"),
    ("why", r"Why\s*:"),
    ("citations", r"(?:Sources|Citations)\s*:"),
    ("clarifying", r"(?:Clarifying Questions|Follow-up questions?|Follow-up)\s*:"),
    ("assumptions", r"Assumptions\s*:"),
]


def parse_sections(text: str) -> ParsedSections:
    text = text or ""
    positions = []
    for key, pattern in _HEADERS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            positions.append((m.start(), key, m.end()))
    positions.sort(key=lambda x: x[0])
    slices: Dict[str, str] = {k: "" for k, _ in _HEADERS}
    for i, (start, key, end) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[end:next_start].strip()
        if key == "answer":
            slices["answer"] = body
        elif key == "why":
            slices["why"] = body
        elif key == "citations":
            slices["citations"] = body
        elif key == "clarifying":
            slices["clarifying"] = body
        elif key == "assumptions":
            slices["assumptions"] = body
    if positions:
        first_start = positions[0][0]
        if first_start > 0:
            leading = text[:first_start].strip()
            if leading and not slices["answer"]:
                slices["answer"] = leading
    if not any(slices.values()):
        return ParsedSections(
            answer_or_plan=text.strip(),
            why="",
            citations="",
            clarifying_questions="",
            assumptions="",
            raw=text,
        )
    return ParsedSections(
        answer_or_plan=slices["answer"],
        why=slices["why"],
        citations=slices["citations"],
        clarifying_questions=slices["clarifying"],
        assumptions=slices["assumptions"],
        raw=text,
    )
