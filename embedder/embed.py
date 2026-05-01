import json
from dataclasses import dataclass
from pathlib import Path

from chunker.chunk import Chunk

DEFAULT_BATCH_SIZE = 64


@dataclass
class Embedding:
    chunk_id: str
    source: str
    text: str
    vector: list[float]
    model: str
    dim: int


def embed_chunk(chunk: Chunk, embedder) -> Embedding:
    [vector] = embedder([chunk.text])
    return Embedding(
        chunk_id=chunk.id,
        source=chunk.source,
        text=chunk.text,
        vector=vector,
        model=embedder.model,
        dim=embedder.dimension,
    )


def embed_chunks(
    input_dir: Path | str,
    embedder,
    data_store,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Embedding]:
    input_dir = Path(input_dir)

    all_embeddings: list[Embedding] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jsonl":
            continue

        # rebuild Chunks from disk; stages must not call into each other's code
        chunks: list[Chunk] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))

        # batch the embedder call: one HTTP request per batch for the API backend,
        # better CPU/GPU utilization for the local backend
        embeddings: list[Embedding] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = embedder([c.text for c in batch])
            for c, v in zip(batch, vectors):
                embeddings.append(Embedding(
                    chunk_id=c.id,
                    source=c.source,
                    text=c.text,
                    vector=v,
                    model=embedder.model,
                    dim=embedder.dimension,
                ))

        data_store.add(embeddings=embeddings, source_stem=path.stem)
        all_embeddings.extend(embeddings)

    return all_embeddings
