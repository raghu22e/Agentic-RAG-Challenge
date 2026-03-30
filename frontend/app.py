"""Flask UI: chat, retrieved chunks, citations, course plan sections, reasoning steps."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_bootstrap import load_project_dotenv

load_project_dotenv(ROOT)

from agents.orchestrator import CoursePlanningOrchestrator
from agents.parse_response import parse_sections
from embeddings.embedder import Embedder
from retriever.catalog_retriever import CatalogRetrieverAgent
from src.config import DATA_DIR, DATA_ROOT, RAG_FAST, VECTOR_DIR
from src.connection_errors import connection_refused_payload, is_connection_refused
from src.ingestion import ingest_all, pdf_file_count
from vectorstore.faiss_store import FAISSCatalogStore

app = Flask(__name__)
app.secret_key = "course-planning-dev-secret-change-in-production"

_orch_lock = threading.Lock()
_orchestrator: CoursePlanningOrchestrator | None = None
_dotenv_mtime: float | None = None


def _reload_env_if_dotenv_changed() -> None:
    """If ``.env`` was edited, reload it (override env) and drop cached agents so the new API key is used."""
    global _dotenv_mtime
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        m = env_path.stat().st_mtime
    except OSError:
        return
    if _dotenv_mtime == m:
        return
    _dotenv_mtime = m
    load_project_dotenv(ROOT, override=True)
    invalidate_orchestrator()


def _build_orchestrator() -> CoursePlanningOrchestrator:
    """Load embedding model + FAISS once (heavy); reuse across requests."""
    embedder = Embedder()
    store = FAISSCatalogStore(VECTOR_DIR, embedder)
    retriever = CatalogRetrieverAgent(store)
    return CoursePlanningOrchestrator(retriever)


def get_orchestrator() -> CoursePlanningOrchestrator:
    global _orchestrator
    _reload_env_if_dotenv_changed()
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = _build_orchestrator()
    return _orchestrator


def invalidate_orchestrator() -> None:
    """Call after reindex/upload so the next request reloads index + embedder."""
    global _orchestrator
    with _orch_lock:
        _orchestrator = None


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def _api_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found", "path": request.path}), 404
    return e


@app.post("/api/chat")
def api_chat():
    t0 = time.perf_counter()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message", "timing_ms": _elapsed_ms()}), 400

        profile = session.get("profile") or {}
        dialogue = session.get("dialogue") or []
        orch = get_orchestrator()
        result = orch.run_turn(message, profile, dialogue=[tuple(x) for x in dialogue])
        session["profile"] = result.profile
        dialogue.append([message, result.reply_text])
        session["dialogue"] = dialogue[-20:]

        parsed = (
            result.parsed if hasattr(result.parsed, "answer_or_plan") else parse_sections(result.reply_text)
        )
        chunks_out = []
        for c in result.retrieved_chunks:
            chunks_out.append(
                {
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                }
            )

        timing_ms = _elapsed_ms()
        steps = list(result.reasoning_steps)
        steps.append(f"Total server time: {timing_ms} ms (RAG_FAST={RAG_FAST})")

        chat_reply = (parsed.answer_or_plan or "").strip() or result.reply_text

        return jsonify(
            {
                "reply": result.reply_text,
                "chat_reply": chat_reply,
                "sections": {
                    "answer_or_plan": parsed.answer_or_plan,
                    "why": parsed.why,
                    "citations": parsed.citations,
                    "clarifying_questions": parsed.clarifying_questions,
                    "assumptions": parsed.assumptions,
                },
                "retrieved_chunks": chunks_out,
                "reasoning_steps": steps,
                "profile": result.profile,
                "timing_ms": timing_ms,
            }
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run: python main.py ingest", "timing_ms": _elapsed_ms()}), 503
    except Exception as e:
        app.logger.exception("api_chat failed")
        if is_connection_refused(e):
            body = connection_refused_payload(str(e))
            body["timing_ms"] = _elapsed_ms()
            return jsonify(body), 503
        return jsonify({"error": str(e) or "Chat failed", "timing_ms": _elapsed_ms()}), 500


@app.post("/api/upload")
def api_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file"}), 400
        f = request.files["file"]
        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "PDF only"}), 400
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        safe = Path(f.filename).name
        path = DATA_DIR / safe
        f.save(str(path))
        docs = ingest_all(DATA_ROOT)
        embedder = Embedder()
        store = FAISSCatalogStore(VECTOR_DIR, embedder)
        store.build(docs)
        invalidate_orchestrator()
        rel = path.relative_to(DATA_ROOT).as_posix()
        return jsonify({"ok": True, "saved_as": rel, "chunks": len(docs)})
    except Exception as e:
        app.logger.exception("api_upload failed")
        return jsonify({"error": str(e)}), 500


@app.post("/api/reindex")
def api_reindex():
    try:
        docs = ingest_all(DATA_ROOT)
        embedder = Embedder()
        store = FAISSCatalogStore(VECTOR_DIR, embedder)
        store.build(docs)
        invalidate_orchestrator()
        return jsonify({"ok": True, "chunks": len(docs), "pdf_files": pdf_file_count(DATA_ROOT)})
    except Exception as e:
        app.logger.exception("api_reindex failed")
        return jsonify({"error": str(e)}), 500


@app.get("/api/health")
def health():
    from src.llm_provider import describe_llm_backend

    _reload_env_if_dotenv_changed()
    llm = describe_llm_backend()
    return jsonify(
        {
            "ok": True,
            "data_root": str(DATA_ROOT),
            "upload_target": str(DATA_DIR),
            "pdf_files": pdf_file_count(DATA_ROOT),
            "index_dir": str(VECTOR_DIR),
            "rag_fast": RAG_FAST,
            **llm,
        }
    )


if __name__ == "__main__":
    # threaded=True: first request warms cache; others can still connect
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
