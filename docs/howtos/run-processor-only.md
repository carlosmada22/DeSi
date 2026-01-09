# Run Processor Only

Convert scraped Markdown into cleaned chunks and persist them to ChromaDB.

## Command
```bash
python -m desi.processor.cli [options]
```

### Arguments
- `--dswiki-input` (default `./data/raw/wikijs/daily`): source directory for Wiki.js Markdown.
- `--openbis-input` (default `./data/raw/openbis/improved`): source directory for openBIS Markdown.
- `--output-dir` (default `./data/processed`): base folder for exported chunks (CSV/JSON/JSONL per source).
- `--chroma-dir` (default `./desi_vectordb`): Chroma persistence directory.
- `--no-delete`: append to existing Chroma data instead of deleting it first.
- `--verbose`: enable DEBUG logs.

`DsWikiProcessor` and `OpenBisProcessor` run sequentially. Each enriches metadata, chunks content, exports artifacts, and writes embeddings via `OllamaEmbeddings` into the same Chroma collection.

## Examples
- Process existing raw data into a fresh vector store:
  ```bash
  python -m desi.processor.cli --dswiki-input data/raw/wikijs --openbis-input data/raw/openbis --chroma-dir desi_vectordb
  ```
- Keep the current database and add new chunks:
  ```bash
  python -m desi.processor.cli --no-delete
  ```

If either input directory is missing, the CLI exits with an error—run the scraper first or adjust paths.
