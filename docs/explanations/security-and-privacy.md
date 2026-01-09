# Security and Privacy

DeSi is designed to run locally; no external APIs are contacted by default.

- **Local storage**: scraped Markdown (`data/raw/**`), processed exports (`data/processed/**`), embeddings (`desi_vectordb/`), and chat history (`data/conversation_memory.db`) live on disk.
- **Models**: Ollama serves models locally; ensure you trust the model weights you pull.
- **Web API**: FastAPI endpoints are unauthenticated by default. Bind to `127.0.0.1` for local-only access or place behind your own reverse proxy/auth if exposing outside the workstation.
- **Secrets**: `DESI_SECRET_KEY` and `DESI_CORS_ORIGINS` default to permissive values; override them in production.
- **Data minimization**: delete `desi_vectordb/` or `data/conversation_memory.db` to purge stored content and history. Re-run processors to rebuild the store.

Always review `.env` contents and CLI flags before deploying beyond a personal machine.
