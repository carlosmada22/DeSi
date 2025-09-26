import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.scraper.readthedocs_scraper import ReadTheDocsScraper

# The main URL of the wiki you want to scrape
BASE_URL = "https://openbis.readthedocs.io/en/20.10.0-11/index.html"

# A directory to save the scraped files
OUTPUT_DIR = "../data/raw/openbis"

# Limit the number of pages for this test run
MAX_PAGES = 10

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("--- Starting ReadTheDocs Scraper ---")
    print(f"Base URL: {BASE_URL}")
    print(f"Saving output to: '{OUTPUT_DIR}'")

    try:
        # Initialize the scraper with our configuration
        scraper = ReadTheDocsScraper(
            base_url=BASE_URL, output_dir=OUTPUT_DIR, max_pages=MAX_PAGES
        )

        # Run the main scraping process
        scraper.scrape()

        print("\n--- Scraping test complete! ---")

    except Exception as e:
        logging.error(
            f"An error occurred during the scraping process: {e}", exc_info=True
        )
        print("\nAn error occurred. Check the logs for details.")
