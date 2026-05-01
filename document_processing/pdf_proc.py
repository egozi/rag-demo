from pathlib import Path

import pymupdf


def extract_pdf_basic(path: Path | str) -> tuple[str, dict]:
    path = Path(path)
    with pymupdf.open(str(path)) as pdf:
        text = "\n\n".join(page.get_text() for page in pdf)
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

    return text, {"title": title, "authors": authors}
