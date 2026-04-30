# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

A minimal, modular RAG demo, modeled loosely on paper-qa. **Stages 1 (document processing) and 2 (chunker)** are implemented. Remaining stages, planned but not yet built: embedder → vector store → retriever → answerer.

The goal is for each stage to be added, swapped, or upgraded independently. New work should preserve that property — favor a new sibling module over coupling stages together.

## Architecture: folder-contract pipeline

Stages communicate primarily through `data/` subfolders. Each stage reads files from one `data/<input>/` folder, writes files to one `data/<output>/` folder, and can be run, inspected, or replaced in isolation.

Current and planned layout:

```
data/
  raw/         # stage 1 input  — PDFs, .md, .txt dropped here by the user
  markdown/    # stage 1 output — one .md per source file, with YAML frontmatter (title, authors, source)
  chunks/      # stage 2 output — one .jsonl per source doc, one chunk per line
  embeddings/  # planned: stage 3 output — parquet with chunk_id, source, text, vector
  index/       # planned: stage 4 — serialized vector store
```

`data/` is gitignored. The notebook `mkdir`s the dirs it needs on startup. New stages should do the same — never assume a folder exists at import time.

**On-disk format matters.** Stage 1 writes `.md` with YAML frontmatter so the file is self-contained — stage 2 can read it back via `python-frontmatter` and reconstruct a full `Document` (text + metadata) without re-running stage 1. Future stages should keep this self-contained property when designing their on-disk format.

**Cross-stage imports — types yes, behavior no.** A stage may import the *type* defined by an earlier stage (e.g. `chunker` imports `Document` from `document_processing.extract` so its pure function can take it as input). What it must not do is import another stage's *functions* — never call `extract_doc` from inside the chunker. Chain stages from the notebook (or a future top-level orchestrator), not from inside another stage.

## Per-stage module shape

Each stage is a **top-level package at the repo root** (e.g. `document_processing/`, `chunker/`), not nested under a parent package. The notebook adds `ROOT` to `sys.path`, so this just works.

Inside each stage package, the pure function and the batch helper live in **one file named after the stage's verb** (`extract.py`, `chunk.py`):

- **Pure single-item function** — e.g. `extract_doc(path) -> Document`, `chunk_doc(doc) -> list[Chunk]`. No I/O beyond reading the input. Notebook-friendly.
- **Batch helper** in the same file — e.g. `extract_docs(input_dir, output_dir) -> list[Document]`, `chunk_docs(input_dir, output_dir) -> list[Chunk]`. Iterates the input folder, calls the pure function, writes outputs to disk.

Naming pattern: singular for the pure function (`<verb>_doc`), plural for the batch helper (`<verb>_docs`). Both are re-exported from the package's `__init__.py`. Do not introduce a separate `batch.py` — the file name should describe what the stage does, not how it's organized.

**Types live with their producer.** `Document` is defined in `document_processing/extract.py`; `Chunk` is in `chunker/chunk.py`. Don't add a shared `types.py`. Each stage owns its output type.

`Document.metadata` is an open dict — stage 1 populates `{"title": str | None, "authors": list[str]}` from the PDF's embedded metadata (with cleanup for empty/junk values), and later stages may enrich it.

## Coding conventions

- **Imports at the top of every file.** No imports inside functions, even for heavy/optional dependencies — declare them at module scope.
- **Keep the code flat.** Don't introduce helper functions just to split a small dispatch. For example, `extract_doc()` handles its PDF/text branches inline with `if/elif`; do not refactor that into per-format helpers like `_extract_pdf()` / `_extract_text()`. Prefer one obvious function over a small tree of trivial ones.

## Running the demo

```bash
pip install -r requirements.txt jupyter
jupyter notebook notebooks/01_demo.ipynb
```

Drop a PDF (or `.md`/`.txt`) into `data/raw/` before running. The notebook is meant to be a single end-to-end walkthrough — when a new stage lands, extend this notebook rather than creating a parallel one (per-stage notebooks may come later).

There is no `pyproject.toml`, test suite, or linter configured yet. Add them only when there's a concrete reason to.
