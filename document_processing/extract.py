from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
import pymupdf
import pymupdf4llm

PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES


@dataclass
class Document:
    source: Path
    text: str
    metadata: dict = field(default_factory=dict)


def extract_doc(path: Path | str) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()

    title: str | None = None
    authors: list[str] = []

    if suffix in PDF_SUFFIXES:
        text = pymupdf4llm.to_markdown(str(path))
        pdf = pymupdf.open(str(path))
        meta = pdf.metadata or {}
        pdf.close()
        raw_title = (meta.get("title") or "").strip()
        if raw_title and raw_title != path.stem and not raw_title.lower().startswith("microsoft word - "):
            title = raw_title
        raw_author = (meta.get("author") or "").strip()
        if ";" in raw_author:
            authors = [a.strip() for a in raw_author.split(";") if a.strip()]
        elif raw_author:
            authors = [a.strip() for a in raw_author.split(",") if a.strip()]
    elif suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")

    return Document(
        source=path,
        text=text,
        metadata={"title": title, "authors": authors},
    )


def extract_docs(input_dir: Path | str, output_dir: Path | str) -> list[Document]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc = extract_doc(path)
        post = frontmatter.Post(
            doc.text,
            source=doc.source.name,
            title=doc.metadata["title"],
            authors=doc.metadata["authors"],
        )
        (output_dir / f"{path.stem}.md").write_text(frontmatter.dumps(post), encoding="utf-8")
        docs.append(doc)
    return docs
