from api.config import get_settings
from embedder.models import ApiEmbedder, LocalEmbedder, OllamaEmbedder


def get_embedder() -> ApiEmbedder | OllamaEmbedder | LocalEmbedder:
    settings = get_settings()
    if settings.llm_backend == "openai":
        return ApiEmbedder(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        )
    elif settings.llm_backend == "huggingface":
        return LocalEmbedder(model=settings.hf_embed_model)
    else:
        return OllamaEmbedder(
            model=settings.ollama_embed_model,
            host=settings.ollama_host,
        )
