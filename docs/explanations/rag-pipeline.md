# RAG Pipeline

`RAGQueryEngine` (`src/desi/query/query.py`) orchestrates retrieval and generation.

1. **Load prompt template** from `prompts/desi_query_prompt.md`.
2. **Connect to ChromaDB** at `DESI_DB_PATH` using `OllamaEmbeddings` (`DESI_EMBEDDING_MODEL_NAME`, default `nomic-embed-text`).
3. **Retrieve**: `similarity_search_with_relevance_scores` fetches a candidate pool (`top_k * 4`), filters scores below `relevance_score_threshold` (default `0.3` in code; CLI defaults to `0.7`), boosts `dswiki` chunks by `dswiki_boost` (0.15 default), and returns the top-k documents.
4. **Prompt build**: merges conversation history, chunk metadata (origin/source), and the user query into the template.
5. **Generate**: invokes `ChatOllama` with `DESI_MODEL_NAME` (code default `gpt-oss:20b`, CLI default `qwen3`), strips `<think>` blocks, and returns the cleaned answer plus the chunks used.

Relevance tuning:
- Increase `--relevance-threshold` to demand higher similarity (fewer, more precise chunks).
- Adjust `DESI_RETRIEVAL_TOP_K` or CLI `top_k` to change context breadth.
- `dswiki_boost` biases results toward DataStore content when scores tie.
