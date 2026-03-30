"""Load project-root ``.env`` (SARVAM_API_KEY, OPENAI_API_KEY, etc.); never log secrets."""
from __future__ import annotations

from pathlib import Path


def load_project_dotenv(project_root: Path, *, override: bool = False) -> bool:
    """
    Load ``project_root / .env`` if python-dotenv is installed.
    ``override=True`` re-applies values from the file (use after the user edits ``.env``).
    Returns True if a .env file exists at that path (even if empty).
    """
    env_path = project_root / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return env_path.is_file()

    if not env_path.is_file():
        return False

    try:
        load_dotenv(env_path, encoding="utf-8-sig", override=override)
    except TypeError:
        try:
            load_dotenv(env_path, override=override)
        except TypeError:
            load_dotenv(env_path)
    return True
