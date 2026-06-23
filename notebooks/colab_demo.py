# %% [markdown]
# # RAG Demo — Google Colab
#
# Runs the full RAG pipeline with **local HuggingFace models on the Colab GPU**.
# No Docker, no Ollama server needed — models download automatically via `transformers`.
#
# **Requirements:**
# - Colab runtime with GPU: Runtime → Change runtime type → T4 GPU
# - An OpenAI API key (used only for RAGAS evaluation judge)
#
# **What this notebook does:**
# 1. Clones the repo and installs Python dependencies
# 2. Configures HuggingFace backend (Phi-3-mini as LLM, MiniLM as embedder)
# 3. Indexes your documents
# 4. Launches the Gradio chat UI with a public share link

# %% [markdown]
# ## Step 1 — Clone the repo and install dependencies

# %%
import os

# Clone the repo (skip if already in the rag-demo directory)
if not os.path.exists("requirements.txt"):
    os.system("git clone https://github.com/YOUR_USERNAME/rag-demo.git")
    os.chdir("rag-demo")

os.system("pip install -q -r requirements.txt")
print("Dependencies installed.")

# %% [markdown]
# ## Step 2 — Configure settings
#
# Set your OpenAI API key below (needed only for RAGAS evaluation judge).
# The LLM and embedder run fully locally on the Colab GPU.
#
# **Default models:**
# - LLM: `microsoft/Phi-3-mini-4k-instruct` (~2.3 GB, fits on T4)
# - Embedder: `sentence-transformers/all-MiniLM-L6-v2` (~80 MB)
#
# To use a different LLM, set `HF_LLM_MODEL` below (must support HuggingFace chat template).
# Examples: `"HuggingFaceH4/zephyr-7b-beta"`, `"google/gemma-2-2b-it"`, `"mistralai/Mistral-7B-Instruct-v0.2"`

# %%
# ── Required ──────────────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk-..."   # ← paste your key here

# ── Colab-specific config ─────────────────────────────────────────────────────
os.environ["LLM_BACKEND"] = "huggingface"
os.environ["QDRANT_PATH"] = "./data/qdrant_local"   # embedded Qdrant, no Docker
os.environ["LANGFUSE_ENABLED"] = "false"            # skip tracing in Colab

# Optional: override the default LLM model
# os.environ["HF_LLM_MODEL"] = "HuggingFaceH4/zephyr-7b-beta"

# Clear cached settings so the env vars above take effect
from api.config import get_settings
get_settings.cache_clear()

settings = get_settings()
print(f"Backend : {settings.llm_backend}")
print(f"LLM     : {settings.hf_llm_model}")
print(f"Embedder: {settings.hf_embed_model}")
print(f"Qdrant  : {settings.qdrant_path}")

# %% [markdown]
# ## Step 3 — Add documents
#
# Upload PDF, `.md`, or `.txt` files to `data/raw/` using the Colab file browser,
# or mount Google Drive and copy files from there.

# %%
import pathlib
pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)

# Optional: mount Google Drive and copy files
# from google.colab import drive
# drive.mount("/content/drive")
# os.system("cp /content/drive/MyDrive/my-papers/*.pdf data/raw/")

files = list(pathlib.Path("data/raw").iterdir())
print(f"Files in data/raw: {[f.name for f in files]}")

# %% [markdown]
# ## Step 4 — Index documents
#
# Downloads the embedding model on first run (~80 MB, cached afterwards).
# Re-run with `--force` if you add new documents.

# %%
os.system("python scripts/index_documents.py --input data/raw --force")

# %% [markdown]
# ## Step 5 — Launch the chat UI
#
# Downloads the LLM on first run (~2.3 GB for Phi-3-mini, cached in `~/.cache/huggingface`).
# Gradio will print a **public share URL** (e.g. `https://xxxx.gradio.live`).
# Share it with your students — it stays active as long as this cell is running.
#
# Ask a question, then click **"Evaluate last answer (RAGAS)"** to see quality scores.

# %%
from api.ui import build_demo

demo = build_demo()
demo.launch(share=True)
