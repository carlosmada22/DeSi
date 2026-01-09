# Troubleshoot

Common issues and quick fixes when running DeSi locally.

- **Ollama not running**: Start the service (`ollama serve`) and ensure the API is reachable at `http://localhost:11434`. Re-run `python init.py` to verify.
- **Missing models**: Pull required models (`ollama pull nomic-embed-text` and your chat model). Mismatched names raise errors when `ChatOllama` or `OllamaEmbeddings` initialize.
- **Empty vector DB**: If answers show “No sources were used,” check that `desi_vectordb/chroma.sqlite3` exists. Rebuild via `python -m desi.processor.cli --force` or delete the directory and reprocess.
- **Scraping failures**: Confirm the target URL is reachable and within the same domain. Set `DESI_MAX_PAGES_PER_SCRAPER` to limit crawl scope. Ensure the `--output` path exists or let the scraper create it.
- **Slow responses**: Use a lighter chat model, reduce `DESI_RETRIEVAL_TOP_K`, or lower `--relevance-threshold` slightly to avoid over-fetching. Hardware limitations on embeddings or model size can dominate latency.
- **Verbose logs**: Set `DESI_LOG_LEVEL=DEBUG` or use `--verbose` on processor/query CLIs to see chunking, retrieval scores, and requests.
- **Stale memory**: Delete `data/conversation_memory.db` or call `POST /api/clear-session` (web) if conversations feel “stuck.”
