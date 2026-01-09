# DeSi Documentation

DeSi (DataStore Helper) is a retrieval-augmented chatbot that answers questions about openBIS and BAM Data Store documentation. It scrapes docs, processes them into embeddings with ChromaDB, and serves both CLI and web chat flows backed by Ollama models.

- Multi-source: OpenBIS ReadTheDocs and DataStore Wiki.js content combined in one vector store.
- Configurable: Environment-first settings with `.env` support and CLI overrides.
- End-to-end pipeline: Scraper → processor → vector DB → RAG query + LangGraph conversation memory.
- Local-first: Runs entirely on your machine; embeddings, cache, and chat history stay local.
- Two interfaces: Terminal chatbot and FastAPI/Gradio web UI.

## Quickstart
Follow the step-by-step guide in [Tutorials](tutorials/index.md):
1) Install dependencies (`pip install -e ".[dev]"`), ensure Ollama is running with `qwen3` and `nomic-embed-text`.
2) Run `python init.py` to create the `.env` and data folders.
3) Start the full workflow with `python main.py` or launch the web UI via `python -m desi.web.cli --reload`.

## Choose Your Path
- [Tutorials](tutorials/index.md): Learn by doing; first run and first query.
- [How-to Guides](howtos/index.md): Task recipes (scrape only, process only, reindex, change models, troubleshoot).
- [Reference](reference/index.md): CLI tables, configuration defaults, file layout, API docs via mkdocstrings.
- [Explanation](explanations/index.md): Why the architecture works, pipeline details, and design trade-offs.

## Build These Docs Locally
```bash
pip install -r requirements-docs.txt
mkdocs serve   # live preview at http://localhost:8000
mkdocs build   # outputs static site to site/
```
