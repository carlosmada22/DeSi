# Command Line

All commands are Python entry points shipped in the repository. Paths are relative to the project root and assume your virtual environment is active (`pip install -e .`).

## Scraper CLI (`python -m desi.scraper.cli`)
Download documentation pages to Markdown using `OpenbisScraper`.

| Argument | Type / Default | Description |
| --- | --- | --- |
| `--url` | required | Base URL to crawl (e.g., `https://openbis.readthedocs.io/en/20.10.0-11/`). |
| `--output` | required | Directory where Markdown files are written. Created if missing. |

Example:
```bash
python -m desi.scraper.cli --url https://openbis.readthedocs.io/en/20.10.0-11/ --output data/raw/openbis
```

## Processor CLI (`python -m desi.processor.cli`)
Chunk scraped Markdown and write embeddings to ChromaDB.

| Argument | Type / Default | Description |
| --- | --- | --- |
| `--dswiki-input` | str, `./data/raw/wikijs/daily` | Input folder for Wiki.js Markdown. |
| `--openbis-input` | str, `./data/raw/openbis/improved` | Input folder for openBIS Markdown. |
| `--output-dir` | str, `./data/processed` | Base folder for processed exports. |
| `--chroma-dir` | str, `./desi_vectordb` | Chroma persistence directory. |
| `--no-delete` | flag | Append to the existing Chroma DB instead of deleting it first. |
| `--verbose` | flag | Enable DEBUG logging. |

Examples:
```bash
python -m desi.processor.cli --dswiki-input data/raw/wikijs --openbis-input data/raw/openbis --chroma-dir desi_vectordb
python -m desi.processor.cli --no-delete --verbose
```

## Query CLI (`python -m desi.query.cli`)
Run the terminal chatbot against an existing vector store.

| Argument | Type / Default | Description |
| --- | --- | --- |
| `--db-path` | str, `./desi_vectordb` | Chroma persistence directory. |
| `--prompt-template` | str, `./prompts/desi_query_prompt.md` | Prompt template file for the RAG engine. |
| `--memory-db-path` | str, `./data/conversation_memory.db` | SQLite database for conversation history. |
| `--llm-model` | str, `qwen3` | Ollama model for generation and rewriting. |
| `--relevance-threshold` | float, `0.7` | Minimum similarity score to keep a chunk. |
| `--verbose` | flag | Enable DEBUG logging. |

Example:
```bash
python -m desi.query.cli --db-path desi_vectordb --prompt-template prompts/desi_query_prompt.md --relevance-threshold 0.5
```

## Web CLI (`python -m desi.web.cli`)
Launch the FastAPI + Gradio interface via uvicorn.

| Argument | Type / Default | Description |
| --- | --- | --- |
| `--host` | str, config default `0.0.0.0` | Bind address; CLI overrides config. |
| `--port` | int, config default `5000` | Port to listen on. |
| `--reload` | flag | Auto-reload in development. |
| `--config` | path, optional | `.env` file to load before environment variables. |

Example:
```bash
python -m desi.web.cli --host 127.0.0.1 --port 7860 --reload
```

## Main Pipeline (`python main.py`)
Orchestrates scraping, processing, and chat in one command.

| Argument | Type / Default | Description |
| --- | --- | --- |
| `--web` | flag | Start the web interface instead of the terminal chat. |
| `--skip-scraping` | flag | Skip scraping even if no raw data exists. |
| `--skip-processing` | flag | Skip processing and go straight to chat. |
| `--force-scraping` | flag | Re-run scraping even if raw data exists. |
| `--force-processing` | flag | Re-run processing even if Chroma exists. |
| `--config` | path, optional | `.env` file to load. Environment variables still override. |

Examples:
```bash
python main.py --force-scraping --force-processing
python main.py --web --config staging.env
```
