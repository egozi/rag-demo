from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embedder.embed import Embedding
from retriever.retrieve import RetrievalResult


@dataclass
class QdrantStore:
    host: str = "localhost"
    port: int = 6333
    collection: str = "rag_demo"
    _client: QdrantClient | None = None

    def __post_init__(self):
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)

    @classmethod
    def from_client(cls, client: QdrantClient, collection: str = "rag_demo") -> "QdrantStore":
        store = cls.__new__(cls)
        store._client = client
        store.collection = collection
        store.host = ""
        store.port = 0
        return store

    def create_collection(self, dim: int, distance: Distance = Distance.COSINE) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=distance),
            )

    def upsert(self, embeddings: list[Embedding]) -> None:
        points = [
            PointStruct(
                id=abs(hash(emb.chunk_id)) % (2**63),
                vector=emb.vector,
                payload={
                    "chunk_id": emb.chunk_id,
                    "source": emb.source,
                    "text": emb.text,
                },
            )
            for emb in embeddings
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievalResult]:
        result = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )
        return [
            RetrievalResult(
                chunk_id=h.payload["chunk_id"],
                source=h.payload["source"],
                text=h.payload["text"],
                score=h.score,
            )
            for h in result.points
        ]
