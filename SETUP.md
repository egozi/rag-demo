# RAG Demo — Setup Guide

## Overview

The final demo runs five services:

| Service | Port | Role |
|---------|------|------|
| `app` | 8000 | FastAPI + Gradio chat UI |
| `qdrant` | 6333 | Vector store |
| `ollama` | 11434 | Local LLM + embeddings (AWS GPU VM only) |
| `langfuse` | 3000 | Observability dashboard |
| `langfuse-db` | — | Postgres backend for Langfuse (internal) |

Three backends are supported — change `llm_backend` in `api/config.py` to switch:

| Backend | `llm_backend` | LLM | Embedder | GPU needed? | Use case |
|---|---|---|---|---|---|
| **OpenAI** (default) | `"openai"` | `gpt-4o-mini` | `text-embedding-3-small` | No | Local dev / testing |
| **Ollama** | `"ollama"` | `llama3.2:3b` | `nomic-embed-text` | Optional | AWS/self-hosted GPU VM |
| **HuggingFace** | `"huggingface"` | `Phi-3-mini-4k-instruct` | `all-MiniLM-L6-v2` | Recommended | Colab / no Ollama install |

> **After switching backends, always re-index your documents** (`python scripts/index_documents.py --force`) — embedding dimensions differ between backends.

RAGAS evaluation always uses OpenAI (`gpt-4o`) as judge, regardless of backend.

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

## Option C — AWS Deployment (GPU VM + local models)

Recommended instance: **`g4dn.xlarge`** — 1× T4 GPU (16 GB VRAM), 4 vCPUs, 16 GB RAM.

Access is via **SSH tunnel** — no public ports needed beyond SSH (port 22), which the default security group already allows.

### 1. Prerequisites

- AWS CLI configured (`aws configure`)

### 2. Create an EC2 key pair

```bash
aws ec2 create-key-pair \
  --key-name my-key-pair \
  --query "KeyMaterial" \
  --output text > ~/.ssh/my-key-pair.pem

chmod 400 ~/.ssh/my-key-pair.pem
```

### 3. Get the latest Ubuntu 22.04 AMI ID

```bash
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text
```

### 4. Launch the instance

```bash
aws ec2 run-instances \
  --image-id <AMI_ID_FROM_STEP_3> \
  --instance-type g4dn.xlarge \
  --key-name my-key-pair \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-demo}]' \
  --count 1
```

Get the public IP once running:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=rag-demo" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text
```

### 5. SSH in and install dependencies

```bash
ssh -i ~/.ssh/my-key-pair.pem ubuntu@<PUBLIC_IP>
```

Then run:

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA driver
sudo apt-get update && sudo apt-get install -y nvidia-driver-535
sudo reboot
# SSH back in after reboot, then:

# nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU
nvidia-smi
```

### 6. Clone and configure

```bash
git clone <your-repo> && cd rag-demo
cp .env.example .env
nano .env   # set OPENAI_API_KEY (for RAGAS judge)
```

Switch to Ollama mode in `api/config.py`:
```python
llm_backend: str = "ollama"
```

### 7. Start all services

```bash
docker compose up -d
```

### 8. Pull Ollama models

```bash
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

### 9. Configure Langfuse (one-time)

Open an SSH tunnel from your **local machine**:

```bash
ssh -i ~/.ssh/my-key-pair.pem -L 3000:localhost:3000 ubuntu@<PUBLIC_IP>
```

Then open [http://localhost:3000](http://localhost:3000), register, create a project, and copy the API keys into `.env` on the server. Restart the app to pick them up:

```bash
docker compose restart app
```

### 10. Upload documents and index

```bash
# Copy PDFs from your local machine
scp -i ~/.ssh/my-key-pair.pem my-docs/*.pdf ubuntu@<PUBLIC_IP>:~/rag-demo/data/raw/

# Index on the server
docker compose exec app python scripts/index_documents.py --input data/raw/
```

### 11. Access the app

Open an SSH tunnel that forwards both the app and Langfuse:

```bash
ssh -i ~/.ssh/my-key-pair.pem -L 8000:localhost:8000 -L 3000:localhost:3000 ubuntu@<PUBLIC_IP>
```

Then open in your browser:
- **App:** [http://localhost:8000](http://localhost:8000)
- **Langfuse:** [http://localhost:3000](http://localhost:3000)

### Stopping the instance (to save cost)

```bash
# Get instance ID
aws ec2 describe-instances --filters "Name=tag:Name,Values=rag-demo" \
  --query "Reservations[0].Instances[0].InstanceId" --output text

# Stop (keeps disk, no compute charge)
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# Start again later
aws ec2 start-instances --instance-ids <INSTANCE_ID>
```

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
