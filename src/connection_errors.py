"""Detect connection-refused style errors (e.g. WinError 10061) for clearer API messages."""
from __future__ import annotations

from typing import Any


def _iter_causes(exc: BaseException):
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def is_connection_refused(exc: BaseException) -> bool:
    for e in _iter_causes(exc):
        if isinstance(e, ConnectionError):
            return True
        errno = getattr(e, "errno", None)
        if errno in (61, 10061, 111):  # POSIX 111 ECONNREFUSED; Win 10061
            return True
        winerror = getattr(e, "winerror", None)
        if winerror == 10061:
            return True
        msg = str(e).lower()
        if "actively refused" in msg or "connection refused" in msg or "10061" in msg:
            return True
    return False


def connection_refused_payload(detail: str) -> dict[str, Any]:
    return {
        "error": "Could not connect to a required network service (connection refused).",
        "detail": detail,
        "hint_openai": "Check SARVAM_API_KEY / OPENAI_API_KEY, firewall, and API reachability. Restart Flask after fixing .env.",
        "hint_demo": "Or set DEMO_MODE=1 for offline placeholder answers (retrieval still works).",
        "hint_flask": "If this error appeared before any chat: ensure Flask is running (`python main.py serve`) and open http://127.0.0.1:5000",
    }
