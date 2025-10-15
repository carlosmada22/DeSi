# Ensure the source directory is in the Python path for imports
# This allows the test to find the scraper module
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Adjust the path if your 'src' directory is located elsewhere relative to the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from desi.scraper.openbis_scraper import scrape_and_find_links

# --- Test Data and Mocks ---

# 1. Create fake HTML content for our mock server responses
FAKE_HTML_HOME = """
<html>
<body>
  <div role="main">
    <h1>Home Page</h1>
    <p>This is the main content.</p>
    <a href="/en/20.10.0-11/details.html">Details Page Link</a>
    <a href="https://example.com">External Link</a>
    <a href="/en/20.10.0-11/">Link to Self</a>
  </div>
  <div id="sidebar">
    <p>Other content we should ignore.</p>
  </div>
</body>
</html>
"""

FAKE_HTML_DETAILS = """
<html>
<body>
  <div role="main">
    <h2>Details Page</h2>
    <p>This is the details page content.</p>
  </div>
</body>
</html>
"""

FAKE_HTML_NO_MAIN = """
<html>
<body>
  <div>
    <p>This page has no main content div.</p>
  </div>
</body>
</html>
"""


# 2. Create a mock for the requests.get function
class MockResponse:
    def __init__(self, content, status_code=200):
        self.content = content.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.RequestException(f"Error {self.status_code}")


@pytest.fixture
def mock_requests_get(mocker):
    """A pytest fixture to mock requests.get with predefined responses."""

    # A map of URLs to their fake HTML content
    url_map = {
        "https://openbis.readthedocs.io/en/20.10.0-11/": MockResponse(FAKE_HTML_HOME),
        "https://openbis.readthedocs.io/en/20.10.0-11/details.html": MockResponse(
            FAKE_HTML_DETAILS
        ),
        "https://openbis.readthedocs.io/en/20.10.0-11/no-main.html": MockResponse(
            FAKE_HTML_NO_MAIN
        ),
        "https://openbis.readthedocs.io/en/20.10.0-11/error.html": MockResponse(
            "Error", 404
        ),
    }

    def mock_get(url):
        # Return the response from our map, or a 404 if not found
        return url_map.get(url, MockResponse("Not Found", 404))

    # Replace the real requests.get with our mock function
    return mocker.patch("requests.get", side_effect=mock_get)


# --- Unit Tests ---


def test_scrape_single_page_success(mock_requests_get, tmp_path):
    """
    Tests that the scraper can download and correctly save a single page.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"

    # Run the scraper on the temporary directory provided by pytest
    with patch("time.sleep", return_value=None):  # Mock sleep to speed up test
        scrape_and_find_links(base_url, str(tmp_path))

    # Assertions
    expected_file = tmp_path / "en_20.10.0-11.md"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "# Home Page" in content
    assert "This is the main content." in content
    assert "Other content we should ignore" not in content  # Should not scrape sidebar


def test_scrape_crawls_to_second_page(mock_requests_get, tmp_path):
    """
    Tests that the scraper follows an internal link found on the first page.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"

    with patch("time.sleep", return_value=None):
        scrape_and_find_links(base_url, str(tmp_path))

    # Assert that both the home page and the details page were scraped and saved
    assert (tmp_path / "en_20.10.0-11.md").exists()
    details_file = tmp_path / "en_20.10.0-11_details.html.md"
    assert details_file.exists()

    details_content = details_file.read_text(encoding="utf-8")
    assert "## Details Page" in details_content
    assert "This is the details page content." in details_content


def test_ignores_external_and_visited_links(mock_requests_get, tmp_path):
    """
    Tests that the scraper does not follow external links or revisit pages.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"

    with patch("time.sleep", return_value=None):
        scrape_and_find_links(base_url, str(tmp_path))

    # The mock is configured to only know about the 'home' and 'details' URLs.
    # If it tries to access example.com, the mock would fail.
    # We can check the call count to ensure it only visited the valid internal links.
    # It should have been called twice: once for home, once for details.
    assert mock_requests_get.call_count == 2


def test_handles_request_exception_gracefully(mock_requests_get, tmp_path):
    """
    Tests that a network error on one page does not stop the entire process.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"
    # Manually add the error URL to the list of pages to visit
    with patch(
        "builtins.set",
        new={base_url, "https://openbis.readthedocs.io/en/20.10.0-11/error.html"},
    ):
        with patch("time.sleep", return_value=None):
            scrape_and_find_links(base_url, str(tmp_path))

    # The scraper should log an error but continue.
    # The successful page should exist, but the error page should not.
    assert (tmp_path / "en_20.10.0-11.md").exists()
    assert not (tmp_path / "en_20.10.0-11_error.html.md").exists()


def test_handles_page_without_main_content(mock_requests_get, tmp_path):
    """
    Tests that no file is created for a page that lacks the main content div.
    """
    url_no_main = "https://openbis.readthedocs.io/en/20.10.0-11/no-main.html"

    with patch("builtins.set", new={url_no_main}):
        with patch("time.sleep", return_value=None):
            scrape_and_find_links(url_no_main, str(tmp_path))

    # The output directory should be empty because the page had no <div role="main">
    # list(tmp_path.iterdir()) will get all files/dirs in the temp path
    assert len(list(tmp_path.iterdir())) == 0
