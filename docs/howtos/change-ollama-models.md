# Change Ollama Models

Swap the generation or embedding models without touching code.

## Configure
Set environment variables (or add to `.env`):
```
DESI_MODEL_NAME=llama3.2:3b          # chat/generation model
DESI_EMBEDDING_MODEL_NAME=nomic-embed-text
```
`DesiConfig` reads these values throughout the pipeline. The query CLI also accepts `--llm-model` for one-off runs.

## Rebuild Embeddings If Needed
If you change the embedding model, rebuild the vector store:
```bash
set DESI_EMBEDDING_MODEL_NAME=bge-m3   # PowerShell example
python -m desi.processor.cli --chroma-dir desi_vectordb
```

## Validate
1. Ensure Ollama has the models: `ollama pull llama3.2:3b` (or your choice).
2. Run a quick query:
   ```bash
   DESI_MODEL_NAME=llama3.2:3b python -m desi.query.cli --db-path desi_vectordb
   ```
3. Check the startup logs for the model names being loaded.
