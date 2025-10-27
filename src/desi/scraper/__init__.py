"""
Scraper module for DeSi.

This module contains scrapers for the following documentation sources:
- ReadTheDocs scraper
"""

# Import the main classes to make them accessible at the package level
from .openbis_scraper import OpenbisScraper

# Define the public API of this package
__all__ = [
    "OpenbisScraper",
]
