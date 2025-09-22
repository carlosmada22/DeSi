#!/usr/bin/env python3
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiJSParser:
    """Parser for Wiki.js HTML content."""
    def __init__(self):
        # --- THE ONLY CHANGE IS HERE: THE CORRECT SELECTOR ---
        self.content_selector = "main.v-main"
        
        self.ignore_selectors = [
            "nav", "footer", "div.v-navigation-drawer", "div.v-app-bar", "div.v-toolbar",
            "div.breadcrumbs", "div.page-actions", "div.page-meta", "div.toc"
        ]

    def extract_content(self, html_content: str, url: str) -> Dict[str, str]:
        soup = BeautifulSoup(html_content, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title Found"
        if title:
            title = re.sub(r'\s*\|.*$', '', title).strip()

        content_element = soup.select_one(self.content_selector)
        content = ""
        if content_element:
            for selector in self.ignore_selectors:
                for element in content_element.select(selector):
                    element.decompose()
            content = content_element.get_text(separator='\n', strip=True)
        else:
            logger.warning(f"Could not find primary content selector '{self.content_selector}' in {url}")

        return {"title": title, "content": content, "url": url}


class WikiJSScraper:
    """Scraper for Wiki.js documentation sites."""
    def __init__(self, base_url: str, output_dir: str, delay: float = 1.0, max_pages: Optional[int] = None):
        self.base_url = self._sanitize_url(base_url)
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[str] = [self.base_url]
        self.parser = WikiJSParser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.domain = urlparse(self.base_url).netloc
        self.driver = self._setup_driver()

    def _setup_driver(self):
        logger.info("Setting up Selenium WebDriver for headless operation...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("WebDriver setup complete.")
            return driver
        except Exception as e:
            logger.error(f"Failed to setup WebDriver: {e}")
            raise

    def scrape(self):
        logger.info(f"Starting to scrape {self.base_url}")
        pages_scraped = 0
        try:
            while self.urls_to_visit and (self.max_pages is None or pages_scraped < self.max_pages):
                url = self.urls_to_visit.pop(0)
                url_clean = urljoin(url, urlparse(url).path)
                if url_clean in self.visited_urls:
                    continue

                logger.info(f"Scraping ({pages_scraped + 1}/{self.max_pages or 'all'}): {url_clean}")
                try:
                    self.driver.get(url_clean)
                    WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, self.parser.content_selector)))
                    html_content = self.driver.page_source
                    self.visited_urls.add(url_clean)

                    content = self.parser.extract_content(html_content, url_clean)
                    if content and content["content"].strip():
                        self._save_content(content)
                        pages_scraped += 1
                    else:
                        logger.warning(f"Content was empty after parsing {url_clean}")

                    soup = BeautifulSoup(html_content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        self._add_url_to_visit(url_clean, link["href"])

                    time.sleep(self.delay)

                except TimeoutException:
                    logger.error(f"Timeout waiting for content on {url_clean}. Skipping.")
                    self.visited_urls.add(url_clean)
                except Exception as e:
                    logger.error(f"Unexpected error scraping {url_clean}: {e}")
                    self.visited_urls.add(url_clean)
        finally:
            self.close()

        logger.info(f"Finished scraping. Scraped {pages_scraped} pages.")
        
    def _add_url_to_visit(self, base_url: str, href: str):
        if not href: return
        abs_url = urljoin(base_url, href)
        url_clean = urljoin(abs_url, urlparse(abs_url).path)

        if urlparse(url_clean).netloc != self.domain: return
        if url_clean in self.visited_urls or url_clean in self.urls_to_visit: return

        skip_patterns = [r'/api/', r'/login', r'\.(pdf|zip|jpg|png|css|js)$']
        if any(re.search(p, url_clean, re.IGNORECASE) for p in skip_patterns): return

        self.urls_to_visit.append(url_clean)

    def _save_content(self, content: Dict[str, str]):
        url_path = urlparse(content["url"]).path
        filename = (url_path.strip("/").replace("/", "_") or "index") + ".txt"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {content['title']}\nURL: {content['url']}\nSource: datastore\n---\n\n{content['content']}")
        logger.info(f"Saved content to {filepath}")

    def _sanitize_url(self, url: str) -> str:
        return f"https://{url}" if not url.startswith(("http://", "https://")) else url.rstrip("/")
        
    def close(self):
        if self.driver:
            logger.info("Closing WebDriver.")
            self.driver.quit()