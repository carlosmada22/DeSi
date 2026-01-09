# Reindex Vector Store

Rebuild ChromaDB when prompts, chunking, or source data change.

## Steps
1. Stop any running chat/web sessions.
2. Remove the existing store (or move it aside):
   ```bash
   rmdir /s /q desi_vectordb   # Windows PowerShell
   # rm -rf desi_vectordb      # macOS/Linux
   ```
3. (Optional) Refresh raw data:
   ```bash
   python -m desi.scraper.cli --url https://openbis.readthedocs.io/en/20.10.0-11/ --output data/raw/openbis
   ```
4. Run processors to recreate embeddings:
   ```bash
   python -m desi.processor.cli --dswiki-input data/raw/wikijs --openbis-input data/raw/openbis --chroma-dir desi_vectordb
   ```
5. Verify the store exists (`desi_vectordb/chroma.sqlite3` should be present) and restart your preferred interface (`python main.py` or `python -m desi.query.cli`).

## Tips
- Use `--no-delete` only when you intentionally want to append to the current store.
- If you change `DESI_COLLECTION_NAME`, reprocess so chunks land in the new collection.
