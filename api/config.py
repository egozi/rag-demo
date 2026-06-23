from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configuration split:
#   .env        — API keys / secrets only (OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY,
#                 LANGFUSE_SECRET_KEY). Copy .env.example → .env and fill these in.
#   config.py   — everything else (model names, hosts, ports, thresholds).
#                 Defaults here work for local dev. Override any field by setting
#                 the matching env var (e.g. export QDRANT_HOST=qdrant in Docker).


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Secrets (set in .env) ─────────────────────────────────────────────────
    openai_api_key: str = ""       # required for RAGAS judge; also LLM/embed in openai mode
    langfuse_public_key: str = ""  # from Langfuse dashboard; leave blank to disable tracing
    langfuse_secret_key: str = ""

    # ── Mode ──────────────────────────────────────────────────────────────────
    llm_backend: str = "openai"    # "openai" | "ollama"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # ── Ollama (only when llm_backend=ollama) ─────────────────────────────────
    ollama_host: str = "http://localhost:11434"  # Docker: http://ollama:11434
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # ── RAGAS judge (always OpenAI, regardless of llm_backend) ───────────────
    ragas_model: str = "gpt-4o"

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_host: str = "http://localhost:3000"  # Docker: http://langfuse:3000
    langfuse_enabled: bool = True

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"  # Docker: qdrant
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_demo"

    # ── Conversation history ──────────────────────────────────────────────────
    history_db: str = "data/history.db"
    short_term_turns: int = 6      # keep last N turns verbatim in the prompt
    long_term_threshold: int = 12  # summarize older turns when total exceeds this


@lru_cache
def get_settings() -> Settings:
    return Settings()
