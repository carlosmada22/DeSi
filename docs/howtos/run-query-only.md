# Run Query Only

Launch the chatbot directly against an existing vector store.

## Command
```bash
python -m desi.query.cli [options]
```

### Arguments
- `--db-path` (default `./desi_vectordb`): Chroma persistence directory to load.
- `--prompt-template` (default `./prompts/desi_query_prompt.md`): prompt template file.
- `--memory-db-path` (default `./data/conversation_memory.db`): SQLite database for conversation history.
- `--llm-model` (default `qwen3` in the CLI; code defaults in `DesiConfig` use `gpt-oss:20b` for the main pipeline).
- `--relevance-threshold` (default `0.7`): minimum similarity score to accept a chunk.
- `--verbose`: enable DEBUG logging.

The CLI wires `RAGQueryEngine`, `SqliteConversationMemory`, and a `ChatOllama` instance for query rewriting. Ollama must be running; otherwise the CLI exits early.

## Examples
- Use a custom vector store and prompt:
  ```bash
  python -m desi.query.cli --db-path ./desi_vectordb --prompt-template ./prompts/desi_query_prompt.md --relevance-threshold 0.5
  ```
- Change the generation model for a session:
  ```bash
  DESI_MODEL_NAME=llama3.2:3b python -m desi.query.cli --llm-model llama3.2:3b
  ```
