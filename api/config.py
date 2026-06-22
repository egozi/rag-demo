from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mode
    llm_backend: str = "openai"  # "openai" | "ollama"

    # OpenAI (always required for RAGAS judge)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Ollama (only when llm_backend=ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # RAGAS judge (always OpenAI)
    ragas_model: str = "gpt-4o"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_enabled: bool = True

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_demo"

    # Conversation history
    history_db: str = "data/history.db"
    short_term_turns: int = 6
    long_term_threshold: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
