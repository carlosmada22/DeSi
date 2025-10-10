import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md  # Import the markdownify library


def scrape_and_find_links(base_url, output_dir):
    """
    Crawls a readthedocs site, converts the main content of each page
    to Markdown, and saves it to .md files.
    """
    to_visit = {base_url}
    visited = set()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        try:
            print(f"Scraping: {current_url}")
            response = requests.get(current_url)
            response.raise_for_status()
            visited.add(current_url)

            soup = BeautifulSoup(response.content, "lxml")

            # --- Save the content as Markdown ---
            main_content = soup.find("div", role="main")
            if main_content:
                path = urlparse(current_url).path
                clean_path = path.strip("/").replace("/", "_")
                if not clean_path:
                    clean_path = "index"
                # Save as a .md file now
                filename = os.path.join(output_dir, f"{clean_path}.md")

                # Convert the HTML content to Markdown
                # The heading_style="ATX" uses # for headers, which is standard.
                text = md(str(main_content), heading_style="ATX")

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)

            # --- Find new links to visit ---
            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(current_url, href)
                if absolute_url.startswith(base_url) and absolute_url not in visited:
                    to_visit.add(absolute_url)

            time.sleep(0.1)

        except requests.RequestException as e:
            print(f"Error scraping {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {current_url}: {e}")


if __name__ == "__main__":
    BASE_URL = "https://openbis.readthedocs.io/en/20.10.0-11/"
    OUTPUT_DIRECTORY = "./data/raw/openbis/improved"

    print("Starting documentation download (as Markdown)...")
    scrape_and_find_links(BASE_URL, OUTPUT_DIRECTORY)
    print("\nDownload complete.")
