import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import frontmatter

from document_processing.extract import Document

DEFAULT_SIZE = 2000
DEFAULT_OVERLAP = 200


@dataclass
class Chunk:
    id: str
    source: str
    index: int
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_doc(doc: Document, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    assert overlap < size, f"overlap ({overlap}) must be less than size ({size})"

    text = doc.text
    step = size - overlap
    pieces: list[str] = []
    i = 0
    while i < len(text):
        pieces.append(text[i:i + size])
        # stop once this window covers the tail; otherwise the next step would
        # emit a sliver chunk that's fully contained in the previous one
        if i + size >= len(text):
            break
        i += step

    source_path = Path(doc.source)
    # fresh dict per chunk so downstream mutations don't leak across siblings or back to doc
    return [
        Chunk(
            id=f"{source_path.stem}#{idx}",
            source=source_path.name,
            index=idx,
            text=p,
            metadata=dict(doc.metadata),
        )
        for idx, p in enumerate(pieces)
    ]


def chunk_docs(
        input_dir: Path | str,
        output_dir: Path | str,
        size: int = DEFAULT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> list[Chunk]:
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []

    # Iterate over markdown files in the input directory
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue

        # rebuild Document from the on-disk markdown; 
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        doc = Document(
            source=path,
            text=post.content,
            metadata={
                "title": post.get("title"),
                "authors": post.get("authors") or [],
            },
        )

        chunks = chunk_doc(doc, size=size, overlap=overlap)

        # Write the chunks to a JSONL file, one chunk per line
        out = output_dir / f"{path.stem}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c)) + "\n")

        all_chunks.extend(chunks)
    return all_chunks
