#!/usr/bin/env python3
"""
ReadtheDocs Scraper

A module for scraping content from ReadtheDocs documentation sites.
This module extracts all textual content from a ReadtheDocs site and saves it
to text files for use in downstream RAG (Retrieval Augmented Generation) pipelines.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReadTheDocsParser:
    """Parser for ReadtheDocs HTML content."""

    def __init__(self):
        """Initialize the parser."""
        self.content_selectors = [
            "div.document",  # Standard ReadtheDocs content div
            "div[role='main']",  # Alternative content div
            "main",  # HTML5 main element
            "article",  # HTML5 article element
            "div.body",  # Another common content div
        ]

        self.ignore_selectors = [
            "div.sphinxsidebar",  # Sidebar
            "footer",  # Footer
            "nav",  # Navigation
            "div.header",  # Header
            "div.related",  # Related links
            "div.breadcrumbs",  # Breadcrumbs
            "div.sourcelink",  # Source link
        ]

    def extract_content(self, html_content: str, url: str) -> Dict[str, str]:
        """
        Extract the main content from a ReadtheDocs HTML page.
        
        Args:
            html_content: The HTML content of the page
            url: The URL of the page
            
        Returns:
            A dictionary containing the title and content of the page
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        title = ""
        # Extract title from h1 if possible, fallback to title tag
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.string
            title = re.sub(r'\s*—.*$', '', title).strip() # Clean up RTD titles
        
        content_element = None
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            logger.warning(f"Could not find main content in {url}")
            return {"title": title, "content": "", "url": url}
        
        for selector in self.ignore_selectors:
            for element in content_element.select(selector):
                element.decompose()
        
        # Use a simpler, more robust text extraction
        content = content_element.get_text(separator='\n', strip=True)
        
        return {"title": title, "content": content, "url": url}

class ReadTheDocsScraper:
    """Scraper for ReadtheDocs documentation sites."""

    def __init__(
        self,
        base_url: str,
        output_dir: str,
        delay: float = 0.5,
        max_pages: Optional[int] = None
    ):
        ### REVISED: Simplified __init__
        self.start_url = base_url
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_pages = max_pages

        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[str] = [self.start_url]
        self.parser = ReadTheDocsParser()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # The base path for ensuring we stay on the same documentation version
        self.base_path = self._get_base_path(self.start_url)
        
        logger.info(f"Initialized scraper for {self.start_url}")
        logger.info(f"Scraping will be contained to base path: {self.base_path}")

    ### NEW: Helper function to correctly identify the base directory
    def _get_base_path(self, url: str) -> str:
        """Gets the base directory of a URL, removing the filename."""
        parsed = urlparse(url)
        # Rebuild the URL without the filename part of the path
        path_parts = parsed.path.split('/')
        if '.' in path_parts[-1]: # Check if the last part is a file
            base_path = '/'.join(path_parts[:-1]) + '/'
        else:
            base_path = parsed.path
            
        return urljoin(url, base_path)

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid for scraping."""
        # Clean fragments (#) from URL for comparison
        url_clean = urljoin(url, urlparse(url).path)

        if url_clean in self.visited_urls:
            return False

        # Ensure we stay within the same documentation version/directory
        if not url_clean.startswith(self.base_path):
            return False

        # Skip common non-content URLs
        skip_patterns = [r'/genindex', r'/search', r'/py-modindex']
        for pattern in skip_patterns:
            if re.search(pattern, url_clean):
                return False

        return True

    def _save_content(self, content: Dict[str, str]) -> None:
        """Save the content to a file."""
        url_path = urlparse(content["url"]).path
        # Make filename safe for all OS
        filename = url_path.strip("/").replace("/", "_").replace(".html", "")
        if not filename:
            filename = "index"
        filepath = self.output_dir / f"{filename}.txt"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {content['title']}\n")
            f.write(f"URL: {content['url']}\n")
            f.write("Source: openbis\n")
            f.write("---\n\n")
            f.write(content["content"])

        logger.info(f"Saved content to {filepath}")
    
    def scrape(self) -> None:
        """Scrape the ReadtheDocs site."""
        logger.info(f"Starting to scrape {self.start_url}")
        pages_scraped = 0

        while self.urls_to_visit and (self.max_pages is None or pages_scraped < self.max_pages):
            url = self.urls_to_visit.pop(0)
            url_clean = urljoin(url, urlparse(url).path) # Normalize URL by removing fragment

            if url_clean in self.visited_urls:
                continue

            logger.info(f"Scraping ({pages_scraped + 1}/{self.max_pages or 'all'}): {url_clean}")

            try:
                response = requests.get(url_clean, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Skipping {url_clean} (Status code: {response.status_code})")
                    self.visited_urls.add(url_clean)
                    continue

                self.visited_urls.add(url_clean)
                content = self.parser.extract_content(response.text, url_clean)
                self._save_content(content)
                pages_scraped += 1

                soup = BeautifulSoup(response.text, "html.parser")
                # Target the main navigation sidebar for high-quality links
                nav_sidebar = soup.find('nav', class_='wy-nav-side')
                if not nav_sidebar:
                    nav_sidebar = soup # Fallback to whole page if no sidebar found

                for link in nav_sidebar.find_all("a", href=True):
                    href = link["href"]
                    # Use the CURRENT page's URL as the base for the join
                    absolute_url = urljoin(url_clean, href)

                    if self._is_valid_url(absolute_url):
                        if absolute_url not in self.urls_to_visit:
                             self.urls_to_visit.append(absolute_url)

                if self.delay > 0:
                    time.sleep(self.delay)

            except requests.RequestException as e:
                logger.error(f"Error scraping {url_clean}: {e}")
                continue

        logger.info(f"Finished scraping. Scraped {pages_scraped} pages.")