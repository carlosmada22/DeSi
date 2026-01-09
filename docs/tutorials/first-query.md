# First Query

Chat with DeSi from the terminal and read the cited sources.

## Start the Chatbot
You need a populated vector store in `desi_vectordb/` (run `python main.py` once to build it). Then start an interactive session:
```bash
python -m desi.query.cli --db-path desi_vectordb --prompt-template prompts/desi_query_prompt.md
```
If you prefer the end-to-end workflow, `python main.py` drops you into the same chat after scraping/processing when needed.

## Ask a Question
Sample prompts:
- `How do I create a new experiment in openBIS?`
- `Where do I find DataStore upload steps?`

The chatbot will:
1. Rewrite the question (LangGraph node) if conversation history exists.
2. Retrieve chunks from Chroma with a relevance threshold (default `0.7` in the CLI).
3. Generate an answer with the configured Ollama model.

## Read the Sources
After each response the CLI prints sources with origin labels (`dswiki` or `openbis`). Expect output similar to:
```
Assistant: ...answer text...
--- Sources Used ---
- Origin: DataStore Wiki, Source: guides/upload.md
- Origin: openBIS Wiki, Source: user_doc/experiments_create.md
```

If you see “No sources were used,” the query did not pass the relevance threshold—try a more specific question or rebuild embeddings with updated data.
