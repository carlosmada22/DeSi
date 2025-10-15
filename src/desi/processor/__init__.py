"""
Unified RAG Processor for Multi-Source Documentation

This package provides a complete pipeline for processing documentation from
multiple sources (e.g., DSWiki, openBIS). It includes specialized processors
for each source and a master CLI to orchestrate the entire workflow.
"""

# Import the main processing function from each specialized processor
# Import the shared data structure, as it's a core concept of the package
from .ds_processor import Document, run_dswiki_processing
from .openbis_processor import run_openbis_processing

# Define the public API of the 'processor' package
__all__ = [
    "run_dswiki_processing",
    "run_openbis_processing",
    "Document",
]
