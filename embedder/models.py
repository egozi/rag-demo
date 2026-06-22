import ollama as ollama_client
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_API_MODEL = "text-embedding-3-small"

# OpenAI doesn't expose a dimension-query endpoint, so hardcode the well-known
# models. Unknown model => KeyError, which is the loud failure we want.
OPENAI_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        self.model = model
        self._st = SentenceTransformer(model)
        self.dimension = self._st.get_sentence_embedding_dimension()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = self._st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


class ApiEmbedder:
    def __init__(self, model: str = DEFAULT_API_MODEL, api_key: str | None = None):
        self.model = model
        # load .env so OpenAI() can pick up OPENAI_API_KEY when no explicit key is passed
        if api_key is None:
            load_dotenv()
        self._client = OpenAI(api_key=api_key)
        self.dimension = OPENAI_MODEL_DIMS[model]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self._client = ollama_client.Client(host=host)
        # Probe dimension by embedding a single token
        probe = self._client.embeddings(model=model, prompt=" ")
        self.dimension = len(probe["embedding"])

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [
            self._client.embeddings(model=self.model, prompt=text)["embedding"]
            for text in texts
        ]
