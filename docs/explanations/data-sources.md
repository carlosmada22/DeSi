# Data Sources

DeSi fuses two documentation sources:

- **openBIS ReadTheDocs**: scraped via `OpenbisScraper` starting from `DESI_OPENBIS_URL` (default `https://openbis.readthedocs.io/en/20.10.0-11/index.html`). Files land in `data/raw/openbis/**`.
- **BAM DataStore Wiki.js**: Markdown exports expected under `data/raw/wikijs/**` (or `data/raw/wikijs/daily` for the processor defaults).

Metadata added during processing:
- `origin`: `openbis` or `dswiki`.
- `source`: relative file path of the chunk.
- `section`: derived from directories (e.g., `user-documentation`) or inferred titles.
- `url`/`title` (openBIS only): reconstructed from filenames to the ReadTheDocs URL.
- `id`: stable chunk identifier (`openbis-...` or `dswiki-...`).

These fields surface in responses so users can see where answers came from.
