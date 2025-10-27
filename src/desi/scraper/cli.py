#!/usr/bin/env python3
"""
Command-line interface for the openbis_scraper.
"""

import argparse
import logging
import sys
from pathlib import Path

# This allows the script to find and import 'openbis_scraper.py'
# assuming both files are in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the specific function from your scraper script
from openbis_scraper import scrape_and_find_links

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Scrape a documentation website using the openbis_scraper."
    )

    parser.add_argument(
        "--url", required=True, help="The base URL of the documentation site to scrape."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The directory to save the scraped Markdown files to.",
    )

    args = parser.parse_args()

    logger.info("Starting documentation download (as Markdown)...")
    logger.info(f"  > Scraping URL: {args.url}")
    logger.info(f"  > Outputting to: {args.output}")

    # Call the original function with the arguments from the command line
    scrape_and_find_links(base_url=args.url, output_dir=args.output)

    logger.info("\nDownload complete.")


if __name__ == "__main__":
    sys.exit(main())
