# Run Scraper Only

Use the scraper CLI to download documentation as Markdown without processing or embedding.

## Command
```bash
python -m desi.scraper.cli --url <base-url> --output <path>
```

### Arguments
- `--url` (required): starting page to crawl (e.g., `https://openbis.readthedocs.io/en/20.10.0-11/`).
- `--output` (required): directory where `.md` files are written (created if missing).

The CLI wraps `OpenbisScraper`, which keeps requests within the base domain and saves each page’s main content as Markdown.

## Examples
- Scrape openBIS docs to the default raw folder:
  ```bash
  python -m desi.scraper.cli --url https://openbis.readthedocs.io/en/20.10.0-11/ --output data/raw/openbis
  ```
- Limit pages via code config: set `DESI_MAX_PAGES_PER_SCRAPER` to cap crawl depth.

## Next Step
Run the processor to chunk and embed the scraped files (see [Run Processor Only](run-processor-only.md)).
