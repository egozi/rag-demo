from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from .pdf_proc import extract_pdf_basic

PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES


@dataclass
class Document:
    source: Path
    text: str
    metadata: dict = field(default_factory=dict)


def write_markedown_doc(doc: Document, output_dir: Path) -> Path:
    post = frontmatter.Post(
            doc.text,
            source=doc.source.name,
            title=doc.metadata["title"],
            authors=doc.metadata["authors"],
    )
    (output_dir / f"{doc.source.stem}.md").write_text(frontmatter.dumps(post), encoding="utf-8")


def extract_text(file_path: Path | str) -> Document:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in PDF_SUFFIXES:
        text, metadata = extract_pdf_basic(path)
    elif suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        metadata = {"title": None, "authors": []}
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")

    return Document(source=path, text=text, metadata=metadata)


def extract_docs(input_dir: Path | str, output_dir: Path | str) -> list[Document]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # Extract the document text and metadata
        doc = extract_text(file_path=path)

        # Write the document as a markdown file with frontmatter
        write_markedown_doc(doc, output_dir)

        docs.append(doc)
    return docs
