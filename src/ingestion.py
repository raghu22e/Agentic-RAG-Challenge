"""Load PDFs from data/ subfolders (policy, courses, programme requirements), chunk for RAG."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_ROOT, LEGACY_PDF_DIR
from src.documents import DocChunk

_SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "node_modules"}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc_type_from_folder(folder_name: str) -> str:
    """Map subfolder name to metadata doc_type (handles spaces, UK/US spelling, common typos)."""
    n = folder_name.lower().replace("-", " ").replace("_", " ")
    # Academic / regulations (typo: polocy, acadamic)
    if (
        "policy" in n
        or "polocy" in n
        or "regulation" in n
        or ("acad" in n and ("pol" in n or "regul" in n))
    ):
        return "policy"
    # Programme / B.Tech eligibility (typo: requriments)
    if (
        "requirement" in n
        or "requriment" in n
        or "requriments" in n
        or "eligib" in n
        or "admission" in n
        or (
            ("program" in n or "programme" in n)
            and ("req" in n or "degree" in n or "b.tech" in n or "btech" in n)
        )
    ):
        return "program_requirements"
    # Course materials (typo: cources)
    if (
        "course" in n
        or "cources" in n
        or "catalog" in n
        or "subject" in n
        or "syllabus" in n
    ):
        return "course_catalog"
    return "general"


def _infer_doc_type_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "policy" in lower or "regulation" in lower:
        return "policy"
    if "program" in lower or "curriculum" in lower or "degree" in lower or "eligibility" in lower:
        return "program_requirements"
    return "course_catalog"


def _iter_data_subfolders(data_root: Path) -> List[Path]:
    if not data_root.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in _SKIP_DIR_NAMES:
            continue
        # Skip legacy only if we handle it separately
        if child.resolve() == LEGACY_PDF_DIR.resolve():
            continue
        out.append(child)
    return out


def collect_pdf_paths(data_root: Path | None = None) -> List[tuple[Path, str]]:
    """
    Return sorted list of (absolute_pdf_path, doc_type).
    Scans each immediate subfolder of data/ recursively for *.pdf.
    Also includes data/pdfs/*.pdf if that legacy folder exists.
    """
    root = data_root or DATA_ROOT
    seen: set[str] = set()
    pairs: List[tuple[Path, str]] = []

    for sub in _iter_data_subfolders(root):
        doc_type = _doc_type_from_folder(sub.name)
        for path in sorted(sub.rglob("*.pdf")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((path, doc_type))

    if LEGACY_PDF_DIR.is_dir():
        for path in sorted(LEGACY_PDF_DIR.glob("*.pdf")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((path, _infer_doc_type_from_filename(path.name)))

    pairs.sort(key=lambda x: x[0].as_posix().lower())
    return pairs


def load_pdf_documents(data_root: Path | None = None) -> List[DocChunk]:
    root = data_root or DATA_ROOT
    docs: List[DocChunk] = []
    pairs = collect_pdf_paths(root)

    for path, doc_type in pairs:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            content = _clean_text(raw)
            if not content:
                continue
            meta = {
                "source": rel,
                "page": i + 1,
                "doc_type": doc_type,
            }
            docs.append(DocChunk(page_content=content, metadata=meta))
    return docs


def chunk_documents(documents: List[DocChunk]) -> List[DocChunk]:
    out: List[DocChunk] = []
    for doc in documents:
        text = doc.page_content
        j = 0
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = _clean_text(text[start:end])
            if chunk:
                h = hashlib.sha256(chunk.encode("utf-8", errors="ignore")).hexdigest()[:12]
                src = doc.metadata.get("source", "x")
                cid = f"{src}-p{doc.metadata.get('page', 0)}-{j}-{h}"
                meta = dict(doc.metadata)
                meta["chunk_id"] = cid
                out.append(DocChunk(page_content=chunk, metadata=meta))
                j += 1
            if end >= len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
    return out


def ingest_all(data_root: Path | None = None) -> List[DocChunk]:
    pages = load_pdf_documents(data_root)
    return chunk_documents(pages)


def pdf_file_count(data_root: Path | None = None) -> int:
    return len(collect_pdf_paths(data_root or DATA_ROOT))
