from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .embed import Embedding


# Data stores are duck-typed: any class exposing `add(embeddings, source_stem)` and
# `load() -> list[Embedding]` plugs into embed_chunks via `data_store=`. New backends go in this file.


class ParquetStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def add(self, embeddings: list[Embedding], source_stem: str) -> Path:
        out = self.directory / f"{source_stem}.parquet"
        pq.write_table(pa.table({
            "chunk_id": [e.chunk_id for e in embeddings],
            "source": [e.source for e in embeddings],
            "text": [e.text for e in embeddings],
            "vector": [e.vector for e in embeddings],
            "model": [e.model for e in embeddings],
            "dim": [e.dim for e in embeddings],
        }), out)
        return out

    def load(self) -> list[Embedding]:
        embeddings: list[Embedding] = []
        for path in sorted(self.directory.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".parquet":
                continue
            table = pq.read_table(source=path)
            for row in table.to_pylist():
                embeddings.append(Embedding(**row))
        return embeddings
