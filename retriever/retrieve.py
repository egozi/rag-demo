from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalResult:
    chunk_id: str
    source: str
    text: str
    score: float


def retrieve(
        query: str,
        embedder,
        data_store,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
    
    [query_vector] = embedder([query])
    embeddings = data_store.load()
    if not embeddings:
        return []

    matrix = np.array([emb.vector for emb in embeddings], dtype=np.float32)
    query_arr = np.array(query_vector, dtype=np.float32)

    # cosine similarity = (matrix @ query) / (|rows of matrix| * |query|)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_arr)
    scores = (matrix @ query_arr) / (matrix_norms * query_norm)

    # argsort ascending, reverse for descending; pick top_k
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        RetrievalResult(
            chunk_id=embeddings[i].chunk_id,
            source=embeddings[i].source,
            text=embeddings[i].text,
            score=float(scores[i]),
        )
        for i in top_idx
    ]
