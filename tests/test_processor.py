#!/usr/bin/env python3
"""
Test Script for Enhanced Multi-Source Documentation Processor

This script tests the new processing pipeline with sample documents
from both OpenBIS and Wiki.js sources.
"""

import json
import logging
import tempfile
from pathlib import Path

from desi.processor.content_normalizer import ContentNormalizer
from desi.processor.enhanced_chunker import EnhancedChunker
from desi.processor.metadata_handler import MetadataHandler
from desi.processor.unified_processor import UnifiedProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sample_processing():
    """Test processing with sample documents from both sources."""

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"

        input_dir.mkdir()

        # Create sample OpenBIS file
        openbis_content = """Title: Python (V3 API) - pyBIS!
URL: https://openbis.readthedocs.io/en/20.10.0-11/software-developer-documentation/apis/python-v3-api.html
Source: openbis
---

Python (V3 API) - pyBIS!

pyBIS is a Python module for interacting with openBIS. pyBIS is designed to be most useful in a Jupyter Notebook or IPython environment, especially if you are developing Python scripts for automatisation.

Dependencies and Requirements

pyBIS relies the openBIS API v3
openBIS version 16.05.2 or newer is required
19.06.5 or later is recommended
pyBIS uses Python 3.6 or newer and the Pandas module

Installation

pip install --upgrade pybis

That command will download install pyBIS and all its dependencies. If pyBIS is already installed, it will be upgraded to the latest version.

General Usage

TAB completition and other hints in Jupyter / IPython

in a Jupyter Notebook or IPython environment, pybis helps you to enter the commands
After every dot . you might hit the TAB key in order to look at the available commands.
if you are unsure what parameters to add to a , add a question mark right after the method and hit SHIFT+ENTER
Jupyter will then look up the signature of the method and show some helpful docstring"""

        openbis_file = (
            input_dir
            / "en_20.10.0-11_software-developer-documentation_apis_python-v3-api.txt"
        )
        with open(openbis_file, "w", encoding="utf-8") as f:
            f.write(openbis_content)

        # Create sample Wiki.js file
        wikijs_content = """Title: Register a Project
URL: https://datastore.bam.de/en/How_to_guides/Register_project
Source: datastore
---

/
How_to_guides
/
Register_project
Register a Project
Last edited by
Ariza de Schellenberger, Angela
06/12/2025

To register a Project, navigate to the Lab Notebook in the left-hand menu, open the drop-down menu, select My Space and click on + New Project. The Project form will open.

Code Requirements

Mandatory for openBIS (*)
Allowed characters: A-Z (uppercase), 0-9, '_' (underscore), '-' (hyphen), and '.' (dot)
Separate words with underscores
Should be meaningful, in English, and between 3-30 characters
Cannot be modified or reused.

Description Requirements

Mandatory for BAM Data Store
Should contain enough detail to be understandable to people outside of the group
Should be in the following format: "English//German"
Should contain 2-50 words.

Steps to Register

Select My Space
Click on + New Project
Enter Code and Description
Review the entries and Save."""

        wikijs_file = input_dir / "en_How_to_guides_Register_project.txt"
        with open(wikijs_file, "w", encoding="utf-8") as f:
            f.write(wikijs_content)

        # Initialize processor
        processor = UnifiedProcessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            min_chunk_size=100,
            max_chunk_size=500,  # Smaller for testing
            generate_embeddings=False,  # Disable for testing
        )

        # Process files
        logger.info("Testing processing pipeline...")
        stats = processor.process(output_format="both")

        # Verify results
        assert stats["files_processed"] == 2, (
            f"Expected 2 files, got {stats['files_processed']}"
        )
        assert stats["total_chunks"] > 0, "No chunks were generated"

        # Check output files exist
        json_file = output_dir / "enhanced_chunks.json"
        csv_file = output_dir / "enhanced_chunks.csv"
        jsonl_file = output_dir / "enhanced_chunks.jsonl"

        assert json_file.exists(), "JSON output file not created"
        assert csv_file.exists(), "CSV output file not created"
        assert jsonl_file.exists(), "JSONL output file not created"

        # Load and inspect JSON output
        with open(json_file, encoding="utf-8") as f:
            chunks = json.load(f)

        # Verify chunk structure
        assert len(chunks) > 0, "No chunks in output"

        sample_chunk = chunks[0]
        required_fields = [
            "id",
            "title",
            "source",
            "content",
            "source_type",
            "section_title",
        ]
        for field in required_fields:
            assert field in sample_chunk, f"Missing required field: {field}"

        # Verify source types
        source_types = set(chunk["source_type"] for chunk in chunks)
        assert "openbis" in source_types, "OpenBIS chunks not found"
        assert "datastore" in source_types, "Datastore chunks not found"

        logger.info(
            f"✅ Test passed! Generated {len(chunks)} chunks from {stats['files_processed']} files"
        )

        # Print sample chunk for inspection
        print("\n" + "=" * 50)
        print("SAMPLE CHUNK")
        print("=" * 50)
        print(f"ID: {sample_chunk['id']}")
        print(f"Title: {sample_chunk['title']}")
        print(f"Source Type: {sample_chunk['source_type']}")
        print(f"Section: {sample_chunk['section_title']}")
        print(f"Content Length: {len(sample_chunk['content'])}")
        print(f"Content Preview: {sample_chunk['content'][:200]}...")
        print("=" * 50)

        return True


def test_individual_components():
    """Test individual components of the processing pipeline."""

    logger.info("Testing individual components...")

    # Test content normalizer
    normalizer = ContentNormalizer()

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("""Title: Test Document
URL: https://example.com/test
Source: test
---

Test Document

This is a test document with some content.

Installation

Follow these steps to install.

Usage

Here's how to use it.""")
        temp_file = Path(f.name)

    try:
        normalized = normalizer.normalize_file(temp_file)
        assert "content" in normalized
        assert "title" in normalized
        assert normalized["title"] == "Test Document"
        logger.info("✅ Content normalizer test passed")

        # Test enhanced chunker
        chunker = EnhancedChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk_content(normalized["content"])
        assert len(chunks) > 0
        logger.info(f"✅ Enhanced chunker test passed - generated {len(chunks)} chunks")

        # Test metadata handler
        metadata_handler = MetadataHandler()
        with open(temp_file) as f:
            content = f.read()
        metadata = metadata_handler.extract_metadata(temp_file, content)
        assert "title" in metadata
        assert "source" in metadata
        logger.info("✅ Metadata handler test passed")

    finally:
        temp_file.unlink()  # Clean up

    return True


def main():
    """Run all tests."""
    print("Running Enhanced Processor Tests")
    print("=" * 50)

    try:
        # Test individual components
        test_individual_components()
        print()

        # Test full processing pipeline
        test_sample_processing()

        print("\n🎉 All tests passed!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
