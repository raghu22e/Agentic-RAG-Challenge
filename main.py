#!/usr/bin/env python3
"""
Entrypoint: optional synthetic PDF generation, ingest/build FAISS, evaluate, or Flask UI.

Loads environment variables from a `.env` file in the project root (if python-dotenv is installed).

Usage:
  python main.py generate-data    # optional: synthetic demo PDFs (legacy)
  python main.py ingest           # index all PDFs under data/<subfolders>/
  python main.py evaluate
  python main.py serve            # requires SARVAM_API_KEY or OPENAI_API_KEY unless DEMO_MODE=1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_bootstrap import load_project_dotenv

load_project_dotenv(ROOT)


def cmd_generate() -> None:
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_synthetic_catalogs.py")], check=True)


def cmd_ingest() -> None:
    from embeddings.embedder import Embedder
    from src.config import DATA_ROOT, VECTOR_DIR
    from src.ingestion import collect_pdf_paths, ingest_all, pdf_file_count
    from vectorstore.faiss_store import FAISSCatalogStore

    n = pdf_file_count(DATA_ROOT)
    if n == 0:
        print(
            "No PDFs found under data/. Add files to:\n"
            "  data/academic_policy/     (policies, regulations)\n"
            "  data/courses/             (per-course or subject PDFs)\n"
            "  data/programme_requirements/  (B.Tech eligibility, degree rules)\n"
            "Or use legacy data/pdfs/*.pdf"
        )
        sys.exit(1)
    paths = collect_pdf_paths(DATA_ROOT)
    print(f"Indexing {n} PDF file(s):")
    for p, dt in paths:
        print(f"  [{dt}] {p.relative_to(DATA_ROOT)}")
    docs = ingest_all(DATA_ROOT)
    print(f"Total chunks: {len(docs)}")
    store = FAISSCatalogStore(VECTOR_DIR, Embedder())
    store.build(docs)
    print(f"FAISS index saved to {VECTOR_DIR}")


def cmd_evaluate() -> None:
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "evaluate.py")], check=True)


def cmd_serve() -> None:
    dot_env = ROOT / ".env"
    has_sarvam = bool((os.environ.get("SARVAM_API_KEY") or "").strip())
    has_openai = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    demo = os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes")
    if not has_sarvam and not has_openai and not demo:
        print("=" * 64)
        if not dot_env.is_file():
            print("No SARVAM_API_KEY or OPENAI_API_KEY — no .env file found.")
            print(f"  1) Copy:  copy .env.example .env   (in folder: {ROOT})")
            print("  2) For Sarvam (India): SARVAM_API_KEY=...  (dashboard.sarvam.ai)")
            print("     Or OpenAI: OPENAI_API_KEY=sk-...")
            print("  3) Restart: python main.py serve")
        else:
            print("SARVAM_API_KEY and OPENAI_API_KEY are both empty after loading .env.")
            print(f"  Check: {dot_env}")
            print("  Use KEY=value with no quotes/spaces around =. Rename .env.txt → .env if needed.")
        print("=" * 64)
    os.chdir(ROOT / "frontend")
    subprocess.run([sys.executable, "app.py"], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Course Planning Assistant")
    p.add_argument("command", choices=["generate-data", "ingest", "evaluate", "serve"])
    args = p.parse_args()
    if args.command == "generate-data":
        cmd_generate()
    elif args.command == "ingest":
        cmd_ingest()
    elif args.command == "evaluate":
        cmd_evaluate()
    else:
        cmd_serve()


if __name__ == "__main__":
    main()
