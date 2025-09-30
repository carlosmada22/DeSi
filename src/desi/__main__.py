#!/usr/bin/env python3
"""
Main entry point for the DeSi scraping pipeline.

This script runs both the ReadTheDocs and Wiki.js scrapers sequentially
to gather all necessary raw data for the RAG system.
"""

import logging
import sys
from pathlib import Path

from desi.scraper.readthedocs_scraper import ReadTheDocsScraper
from desi.scraper.wikijs_scraper import WikiJSScraper
from desi.utils.config import DesiConfig

SRC_PATH = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_PATH))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load configuration
config = DesiConfig()

# --- Scraping Targets ---
# Define the URLs and output directories for each scraper
# We use absolute paths to avoid any confusion about where files are saved.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / config.data_dir / "raw"

SCRAPER_CONFIGS = [
    {
        "name": "openBIS (ReadTheDocs)",
        "scraper_class": ReadTheDocsScraper,
        "url": config.openbis_url,
        "output_dir": DATA_DIR / "openbis",
        "max_pages": config.max_pages_per_scraper,
    },
    {
        "name": "DataStore (Wiki.js)",
        "scraper_class": WikiJSScraper,
        "url": config.wikijs_url,
        "output_dir": DATA_DIR / "wikijs",
        "max_pages": config.max_pages_per_scraper,
    },
]


def main():
    """Runs the full scraping pipeline."""
    logger.info("--- Starting DeSi Scraping Pipeline ---")

    for config in SCRAPER_CONFIGS:
        logger.info(f"--- Starting scraper for: {config['name']} ---")

        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        logger.info(f"URL: {config['url']}")
        logger.info(f"Outputting to: {output_dir.resolve()}")

        try:
            # Initialize the correct scraper class with its config
            scraper = config["scraper_class"](
                base_url=config["url"],
                output_dir=output_dir,
                max_pages=config["max_pages"],
            )

            # Run the scraper
            scraper.scrape()

            logger.info(f"--- Finished scraping {config['name']} ---")

        except Exception:
            logger.error(f"Scraper for {config['name']} failed!", exc_info=True)
            # Decide if you want to stop on failure or continue
            # return 1 # Uncomment to stop the whole process if one scraper fails

    logger.info("--- DeSi Scraping Pipeline Finished Successfully ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
