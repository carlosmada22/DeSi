# Configuration

`DesiConfig` (`src/desi/utils/config.py`) is the source of truth. It loads an optional `.env` file, then environment variables. Defaults below come from code; README examples may differ (for example, README suggests `qwen3` while the code default for `DESI_MODEL_NAME` is `gpt-oss:20b`).

| Variable | Default (code) | Meaning / Example |
| --- | --- | --- |
| `DESI_DB_PATH` | `desi_vectordb` | Chroma persistence directory. |
| `DESI_COLLECTION_NAME` | `desi_docs` | Chroma collection name. |
| `DESI_MEMORY_DB_PATH` | `data/conversation_memory.db` | SQLite chat history file. |
| `DESI_MODEL_NAME` | `gpt-oss:20b` | Ollama chat model (README examples use `qwen3`). |
| `DESI_EMBEDDING_MODEL_NAME` | `nomic-embed-text` | Ollama embedding model. |
| `DESI_DATA_DIR` | `data` | Base data directory for scraping. |
| `DESI_PROCESSED_DATA_DIR` | `data/processed` | Output folder for processed chunks. |
| `DESI_OPENBIS_URL` | `https://openbis.readthedocs.io/en/20.10.0-11/index.html` | Default openBIS ReadTheDocs start URL. |
| `DESI_WIKIJS_URL` | `https://datastore.bam.de/en/home` | Default Wiki.js entry URL. |
| `DESI_MAX_PAGES_PER_SCRAPER` | `None` | Optional limit on pages to crawl (integer). |
| `DESI_MIN_CHUNK_SIZE` | `100` | Minimum characters per chunk. |
| `DESI_MAX_CHUNK_SIZE` | `1000` | Maximum characters per chunk. |
| `DESI_CHUNK_OVERLAP` | `50` | Overlap between chunks (currently used in `DesiConfig`, processors implement their own sizing). |
| `DESI_RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve (used by configs; query CLI uses `top_k=5` internally). |
| `DESI_HISTORY_LIMIT` | `20` | Conversation turns to keep in memory. |
| `DESI_LOG_LEVEL` | `INFO` | Logging level for `setup_logging`. |
| `DESI_WEB_HOST` | `0.0.0.0` | Bind address for web UI. |
| `DESI_WEB_PORT` | `5000` | Port for web UI. |
| `DESI_WEB_DEBUG` | `false` | Enable reload/debug mode. |
| `DESI_SECRET_KEY` | `desi_secret_key_change_in_production` | Flask/FastAPI secret key. |
| `DESI_CORS_ORIGINS` | `*` | Allowed CORS origins. |

Environment variables override values loaded from `--config` files and `.env` files.
