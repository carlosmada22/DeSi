# File Layout

Key locations in the repository (relative to the project root).

- `src/desi/`: Python package source.
  - `scraper/openbis_scraper.py`: OpenBIS ReadTheDocs crawler.
  - `processor/ds_processor.py` and `processor/openbis_processor.py`: chunking and embedding pipelines.
  - `query/query.py`: `RAGQueryEngine`.
  - `query/conversation_engine.py`: LangGraph-based chatbot and SQLite memory.
  - `web/app.py`: FastAPI + Gradio app and Pydantic models; `web/cli.py` for launching.
  - `utils/config.py`: `DesiConfig` environment handling.
- `prompts/desi_query_prompt.md`: prompt template used by the query engine.
- `data/raw/`: scraped Markdown (`openbis/`, `wikijs/`).
- `data/processed/`: exported chunks (`openbis/`, `wikijs/`), CSV/JSON/JSONL.
- `desi_vectordb/`: ChromaDB persistence (created by processors).
- `data/conversation_memory.db`: SQLite conversation memory.
- `main.py`: orchestrated CLI (scrape → process → chat/web).
- `init.py`: setup helper (checks Ollama, creates folders, copies `.env`).
- `assets/`, `src/desi/web/static/`: static web assets (if added).
- `tests/`: test suite.
- `README.md`: original project overview; source material for these docs.
