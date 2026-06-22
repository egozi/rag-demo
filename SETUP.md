# RAG Demo — Setup Guide

## Overview

The final demo runs five services:

| Service | Port | Role |
|---------|------|------|
| `app` | 8000 | FastAPI + Gradio chat UI |
| `qdrant` | 6333 | Vector store |
| `ollama` | 11434 | Local LLM + embeddings (GCP only) |
| `langfuse` | 3000 | Observability dashboard |
| `langfuse-db` | — | Postgres backend for Langfuse (internal) |

Two modes are supported:

- **`LLM_BACKEND=openai`** — OpenAI for LLM + embeddings; easiest for local testing
- **`LLM_BACKEND=ollama`** — Ollama on a GPU VM; used in GCP deployment

RAGAS evaluation always uses OpenAI (`gpt-4o`) as judge, in both modes.

---

## Option A — Local Testing (API models, no GPU needed)

### Prerequisites
- Docker Desktop (or Docker Engine + Compose plugin)
- Python 3.12 + conda env `llm` (or any venv)
- An `OPENAI_API_KEY`

### 1. Clone and install

```bash
git clone <your-repo> && cd rag-demo
pip install -r requirements.txt
```

### 2. Start infrastructure (Qdrant + Langfuse)

```bash
docker compose -f docker-compose.dev.yml up -d
```

Wait ~30 seconds for Langfuse to initialize its database.

### 3. Configure Langfuse (one-time)

1. Open [http://localhost:3000](http://localhost:3000)
2. Register an account (local only — no data leaves your machine)
3. Create a project (e.g., "rag-demo")
4. Go to **Settings → API Keys** and create a key pair

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...          # your key
RAGAS_MODEL=gpt-4o

LANGFUSE_PUBLIC_KEY=pk-lf-...  # from step 3
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_ENABLED=true

QDRANT_HOST=localhost
```

### 5. Add documents and index

Drop one or more PDF, `.md`, or `.txt` files into `data/raw/`, then:

```bash
python scripts/index_documents.py --input data/raw/
```

### 6. Start the app

```bash
uvicorn api.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the Gradio UI loads.

---

## Option B — Local Testing with Ollama (no GCP, local models)

Same as Option A, but with Ollama running locally instead of OpenAI.

### 1. Install Ollama

Download from [https://ollama.com](https://ollama.com) and start the server:

```bash
ollama serve   # runs on http://localhost:11434
```

### 2. Pull models

```bash
bash scripts/pull_models.sh
# or manually:
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Configure `.env`

```bash
LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text

OPENAI_API_KEY=sk-...   # still required for RAGAS judge
RAGAS_MODEL=gpt-4o
```

Everything else is the same as Option A (steps 4–6).

---

## Option C — GCP Deployment (GPU VM + local models)

### 1. Create the VM

```bash
gcloud compute instances create rag-demo \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server \
  --maintenance-policy=TERMINATE
```

### 2. Open firewall ports

```bash
gcloud compute firewall-rules create allow-rag-demo \
  --allow tcp:8000,tcp:3000 \
  --target-tags=http-server \
  --description="RAG demo app and Langfuse"
```

### 3. SSH in and install dependencies

```bash
gcloud compute ssh rag-demo
sudo bash setup/gcp_startup.sh
```

The script installs: NVIDIA driver 535, Docker, nvidia-container-toolkit. It prints next steps when complete.

> **Note:** If you attached `gcp_startup.sh` as a startup script via `--metadata-from-file`, this runs automatically on first boot.

### 4. Clone and configure

```bash
git clone <your-repo> && cd rag-demo
cp .env.example .env
nano .env   # set OPENAI_API_KEY and leave LLM_BACKEND=ollama
```

Set these in `.env` before starting:

```bash
LLM_BACKEND=ollama
OPENAI_API_KEY=sk-...   # for RAGAS judge
```

### 5. Start all services

```bash
docker compose up -d
```

### 6. Pull Ollama models (inside the container)

```bash
bash scripts/pull_models.sh --docker
```

### 7. Configure Langfuse (one-time)

1. Open `http://EXTERNAL_IP:3000` (get IP: `gcloud compute instances list`)
2. Register, create a project, copy API keys into `.env`
3. Restart the app container to pick up the keys:
   ```bash
   docker compose restart app
   ```

### 8. Index documents

Upload your PDFs to `data/raw/` (e.g. via `gcloud compute scp`), then:

```bash
docker compose exec app python scripts/index_documents.py --input data/raw/
```

### 9. Access

- **App:** `http://EXTERNAL_IP:8000`
- **Langfuse:** `http://EXTERNAL_IP:3000`

---

## Running Tests

Tests require no external services — they use in-memory Qdrant and mock all LLM/RAGAS calls.

```bash
# Set LANGFUSE_ENABLED=false so no Langfuse connection is attempted
LANGFUSE_ENABLED=false OPENAI_API_KEY=sk-test pytest tests/ -v
```

Expected output: all tests pass in under 10 seconds.

---

## Re-indexing Documents

To add new documents or change the chunking strategy, re-run the indexer with `--force`:

```bash
python scripts/index_documents.py --input data/raw/ --force
# or in Docker:
docker compose exec app python scripts/index_documents.py --input data/raw/ --force
```

`--force` drops and recreates the Qdrant collection before upserting.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Langfuse shows no traces | Check `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set and match the dashboard |
| `ollama: connection refused` | Ensure `OLLAMA_HOST` points to the right address; in Docker use `http://ollama:11434` |
| RAGAS scores are all 0.0 | Check `OPENAI_API_KEY` is valid and `RAGAS_MODEL=gpt-4o` is set |
| Qdrant search returns nothing | Run `index_documents.py` first — the collection may be empty |
| GPU not detected in Docker | Ensure `nvidia-container-toolkit` is installed and `docker compose` uses `runtime: nvidia` |
