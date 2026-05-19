from pathlib import Path

import pymupdf
import pymupdf4llm
from docling.document_converter import DocumentConverter


def extract_pdf_meta(path: Path | str) -> dict:
    path = Path(path)
    with pymupdf.open(str(path)) as pdf:
        raw_meta = pdf.metadata or {}

    title: str | None = None
    raw_title = (raw_meta.get("title") or "").strip()
    if raw_title and raw_title != path.stem and not raw_title.lower().startswith("microsoft word - "):
        title = raw_title

    authors: list[str] = []
    raw_author = (raw_meta.get("author") or "").strip()
    if ";" in raw_author:
        authors = [a.strip() for a in raw_author.split(";") if a.strip()]
    elif raw_author:
        authors = [a.strip() for a in raw_author.split(",") if a.strip()]

    return {"title": title, "authors": authors}


def extract_pdf_basic(path: Path | str) -> tuple[str, dict]:
    path = Path(path)
    with pymupdf.open(str(path)) as pdf:
        text = "\n\n".join(page.get_text() for page in pdf)
    return text, extract_pdf_meta(path)


def extract_pdf_docling(path: Path | str) -> str:
    result = DocumentConverter().convert(str(path))
    return result.document.export_to_markdown()


def extract_pdf_pymupdf4llm(path: Path | str) -> str:
    return pymupdf4llm.to_markdown(str(path))
