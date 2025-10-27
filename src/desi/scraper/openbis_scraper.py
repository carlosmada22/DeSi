import logging
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logger = logging.getLogger(__name__)


class OpenbisScraper:
    """
    Crawls a readthedocs site, converts the main content of each page
    to Markdown, and saves it to .md files.
    """

    def __init__(self, base_url, output_dir, initial_urls=None):
        """
        Initializes the scraper with the target URL and output directory.

        Args:
            base_url (str): The starting URL of the documentation site.
            output_dir (str): The directory where Markdown files will be saved.
            initial_urls (set, optional): A set of initial URLs to crawl.
                                          Defaults to the base_url.
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.to_visit = initial_urls if initial_urls is not None else {base_url}
        self.visited = set()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def scrape(self):
        """
        Starts the crawling and scraping process.
        """
        while self.to_visit:
            current_url = self.to_visit.pop()
            if current_url in self.visited:
                continue

            try:
                logger.info(f"Scraping: {current_url}")
                response = self._fetch_page(current_url)
                self.visited.add(current_url)

                soup = BeautifulSoup(response.content, "lxml")

                self._save_content_as_markdown(soup, current_url)
                self._find_new_links(soup, current_url)

                time.sleep(0.1)

            except requests.RequestException as e:
                logger.info(f"Error scraping {current_url}: {e}")
            except Exception as e:
                logger.info(f"An unexpected error occurred for {current_url}: {e}")

    def _fetch_page(self, url):
        """
        Fetches the content of a single page.

        Args:
            url (str): The URL to fetch.

        Returns:
            requests.Response: The response object.
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response

    def _save_content_as_markdown(self, soup, url):
        """
        Extracts the main content, converts it to Markdown, and saves it.

        Args:
            soup (BeautifulSoup): The parsed HTML of the page.
            url (str): The URL of the page, used for generating the filename.
        """
        main_content = soup.find("div", role="main")
        if main_content:
            path = urlparse(url).path
            clean_path = path.strip("/").replace("/", "_") or "index"
            filename = os.path.join(self.output_dir, f"{clean_path}.md")

            text = md(str(main_content), heading_style="ATX")

            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

    def _find_new_links(self, soup, current_url):
        """
        Finds all internal links on the page and adds them to the crawl queue.

        Args:
            soup (BeautifulSoup): The parsed HTML of the page.
            current_url (str): The URL of the current page to resolve relative links.
        """
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(current_url, href)
            if (
                absolute_url.startswith(self.base_url)
                and absolute_url not in self.visited
            ):
                self.to_visit.add(absolute_url)


if __name__ == "__main__":
    BASE_URL = "https://openbis.readthedocs.io/en/20.10.0-11/"
    OUTPUT_DIRECTORY = "./data/raw/openbis/improved"

    logger.info("Starting documentation download (as Markdown)...")
    scraper = OpenbisScraper(BASE_URL, OUTPUT_DIRECTORY)
    scraper.scrape()
    logger.info("\nDownload complete.")
