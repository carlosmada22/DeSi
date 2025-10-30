# Ensure the source directory is in the Python path for imports
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest
import requests

# Add the src directory to the path to ensure imports work from the root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.scraper.openbis_scraper import OpenbisScraper

# --- Test Data and Mocks ---

FAKE_HTML_HOME = """
<html><body>
  <div role="main">
    <h1>Home Page</h1><a href="details.html">Details</a>
    <a href="https://example.com">External</a>
  </div>
</body></html>
"""

FAKE_HTML_DETAILS = """
<html><body>
  <div role="main"><h2>Details Page</h2><a href="/">Home</a></div>
</body></html>
"""

FAKE_HTML_NO_MAIN = "<html><body><div>No main content</div></body></html>"


class MockResponse:
    """A mock for the requests.Response object."""

    def __init__(self, content, status_code=200):
        self.content = content.encode("utf-8")
        self.status_code = status_code
        self.url = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.RequestException(f"Error {self.status_code}")


@pytest.fixture
def mock_requests_get(mocker):
    """A pytest fixture to mock requests.get with predefined responses."""
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

    def mock_get(url, timeout=None):
        return url_map.get(url, MockResponse("Not Found", 404))

    # Replace the real requests.get with our mock function
    return mocker.patch("requests.get", side_effect=mock_get)


# --- Unit Tests ---


# By adding `mock_requests_get` to the test signature, we activate the mock.
def test_scrape_single_page_success(mock_requests_get, tmp_path):
    """
    Tests that the scraper can download and correctly save a single page.
    This test will NOT hit the live internet because the mock is active.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"
    scraper = OpenbisScraper(base_url=base_url, output_dir=str(tmp_path), max_pages=1)

    with patch("time.sleep", return_value=None):
        scraper.scrape()

    expected_file = tmp_path / "en_20.10.0-11.md"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "# Home Page" in content
    # The filename should be derived from the path, not the full URL
    assert len(list(tmp_path.iterdir())) == 1


def test_scrape_crawls_to_second_page(mock_requests_get, tmp_path):
    """
    Tests that the scraper follows an internal link found on the first page.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"
    scraper = OpenbisScraper(base_url=base_url, output_dir=str(tmp_path))

    with patch("time.sleep", return_value=None):
        scraper.scrape()

    assert (tmp_path / "en_20.10.0-11.md").exists()
    details_file = tmp_path / "en_20.10.0-11_details.html.md"
    assert details_file.exists()

    details_content = details_file.read_text(encoding="utf-8")
    assert "## Details Page" in details_content
    # It should have called the mock for the base URL and the details URL
    assert mock_requests_get.call_count == 2


def test_scrape_stops_at_max_pages(mock_requests_get, tmp_path):
    """
    Tests that the new max_pages safety feature correctly limits the crawl.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"
    # Set max_pages to 1, even though the first page has a link to a second
    scraper = OpenbisScraper(base_url=base_url, output_dir=str(tmp_path), max_pages=1)

    with patch("time.sleep", return_value=None):
        scraper.scrape()

    # The mock should only have been called ONCE
    assert mock_requests_get.call_count == 1
    # Only one file should have been created
    assert len(list(tmp_path.iterdir())) == 1
    assert (tmp_path / "en_20.10.0-11.md").exists()
    assert not (tmp_path / "en_20.10.0-11_details.html.md").exists()


def test_handles_request_exception_gracefully(mock_requests_get, tmp_path):
    """
    Tests that a network error on one page does not stop the entire process.
    """
    base_url = "https://openbis.readthedocs.io/en/20.10.0-11/"
    error_url = "https://openbis.readthedocs.io/en/20.10.0-11/error.html"
    initial_urls = {base_url, error_url}

    scraper = OpenbisScraper(
        base_url=base_url, output_dir=str(tmp_path), initial_urls=initial_urls
    )

    with patch("time.sleep", return_value=None):
        scraper.scrape()

    # Assert that the successful page was created and the error page was not
    assert (tmp_path / "en_20.10.0-11.md").exists()
    assert not (tmp_path / "en_20.10.0-11_error.html.md").exists()

    # --- FIX: Use assert_has_calls to be more specific and robust ---
    # We verify that it *attempted* to call our initial URLs, regardless of
    # what other URLs it discovered and called later.
    expected_calls = [
        call(base_url, timeout=30),
        call(error_url, timeout=30),
    ]
    mock_requests_get.assert_has_calls(expected_calls, any_order=True)


def test_handles_page_without_main_content(mock_requests_get, tmp_path):
    """
    Tests that no file is created for a page that lacks the main content div.
    """
    no_main_url = "https://openbis.readthedocs.io/en/20.10.0-11/no-main.html"
    scraper = OpenbisScraper(
        base_url="https://openbis.readthedocs.io/en/20.10.0-11/",
        output_dir=str(tmp_path),
        initial_urls={no_main_url},
    )

    with patch("time.sleep", return_value=None):
        scraper.scrape()

    # The mock was called, but no file should have been written.
    assert mock_requests_get.call_count == 1
    assert len(list(tmp_path.iterdir())) == 0
