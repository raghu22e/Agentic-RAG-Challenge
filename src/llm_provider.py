"""Chat LLM: Sarvam AI (default for India) or OpenAI via langchain_openai; optional DEMO_MODE stub."""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage

from src.config import (
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    OPENAI_CHAT_MODEL,
    SARVAM_API_BASE,
    SARVAM_CHAT_MODEL,
)

MISSING_KEY_MESSAGE = (
    "No LLM API key found. Set SARVAM_API_KEY (Sarvam AI — https://dashboard.sarvam.ai/key-management ) "
    "or OPENAI_API_KEY in your `.env` file."
)


def _sarvam_api_key() -> str | None:
    k = (os.environ.get("SARVAM_API_KEY") or "").strip()
    return k or None


def _openai_api_key() -> str | None:
    k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return k or None


def _effective_sarvam_key() -> str | None:
    """Sarvam dashboard key in SARVAM_API_KEY, or mistakenly stored only in OPENAI_API_KEY (sk_s…)."""
    sk = _sarvam_api_key()
    if sk:
        return sk
    ok = _openai_api_key()
    if ok and ok.startswith("sk_s"):
        return ok
    return None


def describe_llm_backend() -> dict:
    """Lightweight summary for /api/health (no model load)."""
    from src.config import PROJECT_ROOT

    env_path = PROJECT_ROOT / ".env"
    try:
        import dotenv as _dotenv  # noqa: F401

        dotenv_installed = True
    except ImportError:
        dotenv_installed = False

    sk, ok = _sarvam_api_key(), _openai_api_key()
    eff_sarvam = _effective_sarvam_key()
    base = {
        "env_file_present": env_path.is_file(),
        "env_file_path": str(env_path),
        "python_dotenv_installed": dotenv_installed,
        "sarvam_key_configured": bool(sk),
        "openai_key_configured": bool(ok),
        "llm_key_configured": bool(eff_sarvam or ok),
        "sarvam_routing": bool(eff_sarvam),
        "openai_key_configured_legacy": bool(ok),
    }

    if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        return {**base, "backend": "demo", "reason": "DEMO_MODE"}
    if eff_sarvam:
        note = {}
        if not sk and ok and ok.startswith("sk_s"):
            note["routing_note"] = "Using OPENAI_API_KEY as Sarvam key (sk_s…); prefer SARVAM_API_KEY in .env."
        return {**base, **note, "backend": "sarvam", "model": SARVAM_CHAT_MODEL, "api_base": SARVAM_API_BASE}
    if ok:
        return {**base, "backend": "openai", "model": OPENAI_CHAT_MODEL}
    return {
        **base,
        "backend": "demo",
        "reason": "llm_key_missing",
        "hint": MISSING_KEY_MESSAGE + " Or set DEMO_MODE=1.",
    }


def _user_preview_from_messages(messages) -> str:
    """
    Find text the human actually typed. Agent prompts may be last (e.g. Verifier's 'DRAFT:...'),
    so we must not use the most recent HumanMessage blindly.
    """
    human_contents = [str(m.content) for m in messages if isinstance(m, HumanMessage)]

    for c in human_contents:
        key = "User request:\n"
        if key in c:
            rest = c.split(key, 1)[1].strip()
            rest = rest.split("\n\n")[0].strip()
            if rest:
                return rest[:500]

    for c in human_contents:
        key = "Latest user message:"
        if key in c:
            after = c.split(key, 1)[1].strip()
            line = after.split("\n")[0].strip()
            if line:
                return line[:500]

    for c in human_contents:
        s = c.lstrip()
        if s.startswith("DRAFT:") or s.startswith("DRAFT\n"):
            continue
        if s.upper().startswith("DRAFT"):
            continue
        t = c.strip()
        if t:
            return t.split("\n")[0][:400]

    return "(See your question in the left chat panel.)"


class DemoLLM:
    """Placeholder when DEMO_MODE or no API key; retrieval still runs in the UI."""

    def __init__(self, note: str | None = None) -> None:
        self.note = note

    def invoke(self, messages):
        preview = _user_preview_from_messages(messages)
        banner = (self.note or MISSING_KEY_MESSAGE).strip()
        text = (
            "Chat reply:\n"
            f"{banner}\n\n"
            "I still searched your PDFs — check the right-hand panel for evidence. "
            "To get real answers here, set SARVAM_API_KEY in `.env` (see https://dashboard.sarvam.ai/key-management ) "
            "or OPENAI_API_KEY for OpenAI.\n\n"
            "Sources:\n"
            "- See \"Retrieved chunks\" in the UI for filenames and text.\n\n"
            "Clarifying Questions:\n"
            "Which program (e.g. B.Tech branch) and semester are you in? That helps me tailor advice.\n\n"
            f"(Demo placeholder — your message: {preview})"
        )
        return AIMessage(content=text)


def _build_chat_openai(*, api_key: str, model: str, base_url: str | None) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    timeout = LLM_TIMEOUT_SEC
    kwargs: dict = {
        "model": model,
        "temperature": LLM_TEMPERATURE,
        "api_key": api_key,
        "max_retries": 1,
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")

    try:
        return ChatOpenAI(**kwargs, timeout=timeout)
    except TypeError:
        return ChatOpenAI(**kwargs, request_timeout=timeout)


def get_chat_model():
    if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        return DemoLLM(note="DEMO_MODE is on.")

    sk = _effective_sarvam_key()
    if sk:
        # Sarvam: OpenAI-compatible /v1/chat/completions + Bearer (see Sarvam quickstart).
        return _build_chat_openai(
            api_key=sk,
            model=SARVAM_CHAT_MODEL,
            base_url=SARVAM_API_BASE,
        )

    ok = _openai_api_key()
    if ok:
        return _build_chat_openai(
            api_key=ok,
            model=OPENAI_CHAT_MODEL,
            base_url=None,
        )

    return DemoLLM(note=MISSING_KEY_MESSAGE)
