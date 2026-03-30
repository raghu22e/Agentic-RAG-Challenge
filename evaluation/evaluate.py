"""
Batch evaluation: citation coverage, substring accuracy, abstention accuracy.
Run from project root: python evaluation/evaluate.py
Uses DEMO_MODE-friendly heuristics (substring checks); with a real LLM, interpret results qualitatively.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_bootstrap import load_project_dotenv

load_project_dotenv(ROOT)

from agents.orchestrator import CoursePlanningOrchestrator
from agents.parse_response import parse_sections
from embeddings.embedder import Embedder
from retriever.catalog_retriever import CatalogRetrieverAgent
from vectorstore.faiss_store import FAISSCatalogStore

from src.config import VECTOR_DIR

ABSTAIN_PHRASES = (
    "do not have enough information",
    "cannot verify",
    "not in the retrieved",
    "not enough information in the document",
    "no excerpts",
    "cannot find",
    "not available in",
    "no information in the provided",
)


def load_items():
    path = Path(__file__).parent / "test_dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"]


def citation_coverage(parsed, chunks) -> bool:
    cite = (parsed.citations or "") + (parsed.answer_or_plan or "")
    sources = {c.source for c in chunks}
    if not chunks:
        return False
    mentioned = re.findall(r"([\w\-]+\.pdf)", cite, re.IGNORECASE)
    if mentioned:
        for m in mentioned:
            ml = m.lower()
            if any(s.lower() == ml or Path(s).name.lower() == ml for s in sources):
                return True
        return False
    blob = cite.lower()
    return any(s.lower() in blob for s in sources)


def substring_accuracy(parsed, expected: list[str]) -> bool:
    if not expected:
        return True
    blob = (parsed.answer_or_plan + "\n" + parsed.why + "\n" + parsed.citations).upper()
    return all(exp.upper() in blob for exp in expected)


def abstention_ok(parsed) -> bool:
    text = (parsed.answer_or_plan or "").lower()
    return any(p in text for p in ABSTAIN_PHRASES)


def main() -> None:
    embedder = Embedder()
    store = FAISSCatalogStore(VECTOR_DIR, embedder)
    try:
        store.load()
    except Exception as e:
        print("Build index first: python main.py ingest", e)
        sys.exit(1)

    orch = CoursePlanningOrchestrator(CatalogRetrieverAgent(store))
    items = load_items()

    cov_hits = 0
    acc_hits = 0
    acc_total = 0
    abst_hits = 0
    abst_total = 0

    for it in items:
        r = orch.run_turn(it["question"], {})
        p = parse_sections(r.reply_text)
        if citation_coverage(p, r.retrieved_chunks):
            cov_hits += 1
        if it["category"] == "not_in_docs":
            abst_total += 1
            if abstention_ok(p):
                abst_hits += 1
        else:
            exp = it.get("expected_substrings") or []
            if not exp:
                continue
            acc_total += 1
            if substring_accuracy(p, exp):
                acc_hits += 1

    n = len(items)
    print("=== Evaluation summary ===")
    print(f"Items: {n}")
    print(f"Citation coverage rate: {cov_hits / n:.3f}")
    if acc_total:
        print(f"Accuracy (gold substrings, excl. not_in_docs): {acc_hits / acc_total:.3f} ({acc_hits}/{acc_total})")
    else:
        print("Accuracy: skipped (no items with non-empty expected_substrings — add gold labels in test_dataset.json)")
    print(f"Abstention accuracy (not_in_docs): {abst_hits / max(abst_total,1):.3f}")


if __name__ == "__main__":
    main()
