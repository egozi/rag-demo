from dataclasses import dataclass, field
import json
from pathlib import Path

from .pdf_proc import extract_pdf_basic

PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES


@dataclass
class Document:
    source: Path
    text: str
    metadata: dict = field(default_factory=dict)


def _frontmatter_value(value: object) -> str:
    """Return a YAML-safe scalar/list using JSON syntax, which YAML accepts."""
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False)


def write_markdown_doc(doc: Document, output_dir: Path) -> Path:
    frontmatter = {
        "source": doc.source.name,
        "title": doc.metadata.get("title"),
        "authors": doc.metadata.get("authors", []),
    }
    header = "\n".join(
        f"{key}: {_frontmatter_value(value)}"
        for key, value in frontmatter.items()
    )
    content = f"---\n{header}\n---\n\n{doc.text}"

    out = output_dir / f"{doc.source.stem}.md"
    out.write_text(content, encoding="utf-8")
    return out


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
        write_markdown_doc(doc=doc, output_dir=output_dir)

        docs.append(doc)
    return docs
