#!/usr/bin/env bash
# Pull Ollama models required for the RAG demo.
# Usage:
#   ./scripts/pull_models.sh                    # pull into local Ollama
#   ./scripts/pull_models.sh --docker           # pull inside the ollama Docker container
set -euo pipefail

CHAT_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

if [[ "${1:-}" == "--docker" ]]; then
    echo "Pulling models inside Docker container 'ollama' ..."
    docker compose exec ollama ollama pull "$CHAT_MODEL"
    docker compose exec ollama ollama pull "$EMBED_MODEL"
else
    echo "Pulling models into local Ollama ..."
    ollama pull "$CHAT_MODEL"
    ollama pull "$EMBED_MODEL"
fi

echo ""
echo "Done. Models pulled:"
echo "  Chat:       $CHAT_MODEL"
echo "  Embeddings: $EMBED_MODEL"
