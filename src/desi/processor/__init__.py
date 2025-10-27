"""
Unified RAG Processor for Multi-Source Documentation

This package provides a complete pipeline for processing documentation from
multiple sources (e.g., DSWiki, openBIS). It includes specialized processors
for each source and a master CLI to orchestrate the entire workflow.
"""

# Import the main processor class from each specialized module
from .ds_processor import Document, DsWikiProcessor
from .openbis_processor import OpenBisProcessor

# Define the public API of the 'processor' package
__all__ = [
    "DsWikiProcessor",
    "OpenBisProcessor",
    "Document",
]
