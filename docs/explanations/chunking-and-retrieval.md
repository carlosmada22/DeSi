# Chunking and Retrieval

Chunking strategies aim to preserve context while staying within embedding limits.

## DSWiki (`DsWikiProcessor`)
- Parses YAML frontmatter when present and keeps metadata.
- Chooses strategy: FAQ-style `<details>` splitting or structure-based splitting by headings.
- Cleans Mermaid/PlantUML snippets into text and strips HTML remnants.
- Enriches metadata (`origin=dswiki`, `section`, `source`, `id`), then filters very short chunks.

## openBIS (`OpenBisProcessor`)
- Normalizes ReadTheDocs artifacts (permalink markers, non-breaking spaces).
- Chunks by headings with `ContentChunker` respecting min/max size (`100` / `1000` chars defaults).
- Adds metadata (`origin=openbis`, reconstructed `url`, `title`, `section`, unique `id`).

## Embeddings and Storage
- Both processors call `OllamaEmbeddings(model="nomic-embed-text")` and persist via `Chroma.from_documents`.
- Chroma collection metadata uses cosine similarity; persisted at `DESI_DB_PATH`.

## Retrieval
- Candidate pool size is `top_k * 4`; chunks below the relevance threshold are discarded.
- `dswiki` chunks receive a positive score boost (`dswiki_boost`) before sorting to favor DataStore content when scores tie.
