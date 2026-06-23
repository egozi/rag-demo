# %% [markdown]
# # RAG Demo — Google Colab
#
# Runs the full RAG pipeline with **local Ollama models on the Colab GPU**.
# No Docker, no cloud account needed beyond Colab itself.
#
# **Requirements:**
# - Colab runtime with GPU: Runtime → Change runtime type → T4 GPU
# - An OpenAI API key (used only for RAGAS evaluation judge)
#
# **What this notebook does:**
# 1. Installs Ollama and pulls `llama3.2:3b` + `nomic-embed-text`
# 2. Installs Python dependencies
# 3. Indexes your documents (upload PDFs to the `data/raw/` folder)
# 4. Launches the Gradio chat UI with a public share link

# %% [markdown]
# ## Step 1 — Install Ollama and pull models
#
# This takes ~5 minutes the first time (model download).
# On subsequent runs the models are cached in `/root/.ollama`.

# %%
import subprocess
import time

# Install Ollama
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

# Start Ollama server in background
ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)  # wait for server to be ready

# Pull models (cached after first run)
subprocess.run(["ollama", "pull", "llama3.2:3b"], check=True)
subprocess.run(["ollama", "pull", "nomic-embed-text"], check=True)

print("Ollama ready.")

# %% [markdown]
# ## Step 2 — Clone the repo and install Python dependencies

# %%
# If running from a fresh Colab session, clone the repo first:
# !git clone https://github.com/YOUR_USERNAME/rag-demo.git
# %cd rag-demo

# Install dependencies
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Dependencies installed.")

# %% [markdown]
# ## Step 3 — Configure settings
#
# Set your OpenAI API key below (needed for RAGAS evaluation).
# All other settings are configured for Colab automatically.

# %%
import os

# ── Required ──────────────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk-..."   # ← paste your key here

# ── Colab-specific config (no server needed) ──────────────────────────────────
os.environ["LLM_BACKEND"] = "ollama"
os.environ["QDRANT_PATH"] = "./data/qdrant_local"   # embedded Qdrant, no Docker
os.environ["LANGFUSE_ENABLED"] = "false"            # skip tracing in Colab

# Clear cached settings so the env vars above take effect
from api.config import get_settings
get_settings.cache_clear()

settings = get_settings()
print(f"Backend : {settings.llm_backend}")
print(f"LLM     : {settings.ollama_model}")
print(f"Embedder: {settings.ollama_embed_model}")
print(f"Qdrant  : {settings.qdrant_path}")

# %% [markdown]
# ## Step 4 — Add documents
#
# Upload PDF, `.md`, or `.txt` files to `data/raw/` using the Colab file browser,
# or mount Google Drive and copy files from there.

# %%
import pathlib
pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)

# Optional: mount Google Drive
# from google.colab import drive
# drive.mount("/content/drive")
# !cp /content/drive/MyDrive/my-papers/*.pdf data/raw/

# List files ready to index
files = list(pathlib.Path("data/raw").iterdir())
print(f"Files in data/raw: {[f.name for f in files]}")

# %% [markdown]
# ## Step 5 — Index documents
#
# Runs the full pipeline: extract → chunk → embed → store in Qdrant.
# Use `--force` to re-index if you add new documents.

# %%
subprocess.run(
    ["python", "scripts/index_documents.py", "--input", "data/raw", "--force"],
    check=True,
)

# %% [markdown]
# ## Step 6 — Launch the chat UI
#
# Gradio will print a **public share URL** (e.g. `https://xxxx.gradio.live`).
# Share it with your students — it stays active as long as this cell is running.
#
# Ask a question, then click **"Evaluate last answer (RAGAS)"** to see quality scores.

# %%
from api.ui import build_demo

demo = build_demo()
demo.launch(share=True)
