# Run Locally

Learn what DeSi creates on disk and how to rerun the workflow safely.

## Directory Layout After Setup
- `data/raw/openbis` and `data/raw/wikijs`: scraped Markdown.
- `data/processed/openbis` and `data/processed/wikijs`: cleaned/chunked exports (CSV/JSON/JSONL).
- `desi_vectordb/`: ChromaDB persistence (vector store used by the query engine).
- `data/conversation_memory.db`: SQLite chat history used by LangGraph memory.
- `prompts/desi_query_prompt.md`: prompt template loaded by the RAG engine.

## Typical Local Run
```bash
# assumes venv is active
python main.py
```

### Control the Workflow
- `--skip-scraping` / `--skip-processing`: skip steps even if prerequisites are missing (will error if the next stage has no data).
- `--force-scraping` / `--force-processing`: rerun stages even when outputs already exist (useful after changing chunking or models).
- `--web`: start the FastAPI/Gradio UI instead of the terminal chat.
- `--config <file>`: load alternate `.env` file; environment variables still override file values.

### Rerunning Safely
- To rebuild the vector store from fresh data, delete or move `desi_vectordb/` and rerun `python main.py --force-scraping --force-processing`.
- To keep the current vector DB but refresh memory, delete `data/conversation_memory.db`.
- To inspect processed data before embedding, open the CSV/JSON files under `data/processed/**`.
