import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.scraper.wikijs_scraper import WikiJSScraper


# The main URL of the wiki you want to scrape
BASE_URL = "https://datastore.bam.de/en/home"

# The directory where you want to save the text files
OUTPUT_DIR = "../data/raw/wikijs"

# (Optional) Limit the number of pages to scrape for a quick test
MAX_PAGES = 10

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Starting scraper for {BASE_URL}")
    print(f"Output will be saved to '{OUTPUT_DIR}'")
    
    try:
        scraper = WikiJSScraper(
            base_url=BASE_URL,
            output_dir=OUTPUT_DIR,
            max_pages=MAX_PAGES 
        )
        scraper.scrape()
        print("\nScraping complete!")
    except Exception as e:
        logging.error(f"An error occurred during the scraping process: {e}")
        print(f"\nAn error occurred. Check the logs for details.")