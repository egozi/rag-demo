#!/usr/bin/env python3
"""Index documents from data/raw/ into Qdrant.

Usage:
    python scripts/index_documents.py [--input PATH] [--force]
"""
import argparse
import sys
from pathlib import Path

# Allow imports from the repo root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import get_settings
from api.embed import get_embedder
from api.qdrant_store import QdrantStore
from chunker.chunk import chunk_docs
from document_processing.extract import extract_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    parser.add_argument(
        "--input",
        default="data/raw",
        help="Directory containing raw documents (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()

    settings = get_settings()
    input_dir = Path(args.input)
    markdown_dir = Path("data/markdown")
    chunks_dir = Path("data/chunks")

    for d in (markdown_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Extracting documents from {input_dir} ...")
    docs = extract_docs(input_dir, markdown_dir)
    print(f"  {len(docs)} document(s) extracted")

    print("Chunking ...")
    chunks = chunk_docs(markdown_dir, chunks_dir)
    print(f"  {len(chunks)} chunk(s) produced")

    print("Loading embedder ...")
    embedder = get_embedder()
    print(f"  Using {type(embedder).__name__} ({embedder.model})")

    store = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
        path=settings.qdrant_path,
    )

    if args.force:
        try:
            store._client.delete_collection(settings.qdrant_collection)
            print(f"  Dropped collection '{settings.qdrant_collection}'")
        except Exception:
            pass

    store.create_collection(dim=embedder.dimension)

    print(f"Embedding and upserting {len(chunks)} chunks ...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder([c.text for c in batch])
        from embedder.embed import Embedding

        embeddings = [
            Embedding(
                chunk_id=c.id,
                source=c.source,
                text=c.text,
                vector=v,
                model=embedder.model,
                dim=embedder.dimension,
            )
            for c, v in zip(batch, vectors)
        ]
        store.upsert(embeddings)
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)}", end="\r")

    print(f"\nDone. Indexed {len(chunks)} chunks from {len(docs)} document(s).")
    print(f"Collection: '{settings.qdrant_collection}' on {settings.qdrant_host}:{settings.qdrant_port}")


if __name__ == "__main__":
    main()
