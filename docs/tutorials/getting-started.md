# Getting Started

This walkthrough installs DeSi, checks prerequisites, and runs the full workflow once.

## Prerequisites
- Python 3.8+ (project is tested with 3.10–3.12).
- [Ollama](https://ollama.com) running locally.
- Models: pull at least `nomic-embed-text` (embeddings) and a chat model (`gpt-oss:20b` is the code default; the README examples use `qwen3`). Set alternatives via `DESI_MODEL_NAME` and `DESI_EMBEDDING_MODEL_NAME`.

## Install
```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows (PowerShell)
# source .venv/bin/activate     # macOS/Linux
pip install -e ".[dev]"
```

## Initialize
Run the helper to create folders and copy `.env.example` to `.env` if missing:
```bash
python init.py
```
The script checks Ollama availability, creates `data/raw`, `data/processed`, and `desi_vectordb`, and verifies imports.

## First Full Run
Start the all-in-one pipeline (scrape → process → chat):
```bash
python main.py
```
What happens:
1. If `desi_vectordb/` is missing, the pipeline scrapes sources and processes them.
2. If raw data exists but no vector DB, only processing runs.
3. If both exist, it skips to the chat interface.

Use `--force-scraping` or `--force-processing` to refresh data, or `--skip-*` flags to bypass stages. Add `--web` to launch the FastAPI/Gradio interface instead of the terminal chat.
