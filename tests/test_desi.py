#!/usr/bin/env python3
"""
Test script for DeSi system.

This script tests various components of DeSi to ensure they work correctly.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.processor.processor import MultiSourceRAGProcessor
from desi.query.query import DesiRAGQueryEngine
from desi.scraper.readthedocs_scraper import ReadTheDocsScraper
from desi.scraper.wikijs_scraper import WikiJSScraper
from desi.utils.logging import setup_logging
from desi.utils.vector_db import DesiVectorDB

# Configure logging
logger = logging.getLogger(__name__)


def test_readthedocs_scraper():
    """Test the ReadTheDocs scraper."""
    logger.info("Testing ReadTheDocs scraper...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            scraper = ReadTheDocsScraper(
                base_url="https://openbis.readthedocs.io/en/20.10.0-11/",
                output_dir=temp_dir,
                max_pages=2,  # Limit for testing
            )

            # Test URL validation
            assert scraper._is_valid_url(
                "https://openbis.readthedocs.io/en/20.10.0-11/user-documentation/general-users/index.html"
            )
            assert not scraper._is_valid_url("https://example.com/")

            logger.info("✓ ReadTheDocs scraper basic functionality works")
            return True

        except Exception as e:
            logger.error(f"✗ ReadTheDocs scraper test failed: {e}")
            return False


def test_wikijs_scraper():
    """Test the Wiki.js scraper."""
    logger.info("Testing Wiki.js scraper...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            scraper = WikiJSScraper(
                base_url="https://datastore.bam.de/en/home",  # Dummy URL for testing
                output_dir=temp_dir,
                max_pages=1,
            )

            # Test URL validation
            assert scraper._is_valid_url("https://datastore.bam.de/en/How_to_guides")
            assert not scraper._is_valid_url("https://other.com/page1")

            logger.info("✓ Wiki.js scraper basic functionality works")
            return True

        except Exception as e:
            logger.error(f"✗ Wiki.js scraper test failed: {e}")
            return False


def test_processor():
    """Test the multi-source processor."""
    logger.info("Testing multi-source processor...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test input files
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()

            # Create test file with openBIS source
            test_file1 = input_dir / "test1.txt"
            with open(test_file1, "w", encoding="utf-8") as f:
                f.write("Title: Test Document 1\n")
                f.write("URL: https://example.com/test1\n")
                f.write("Source: openbis\n")
                f.write("---\n\n")
                f.write(
                    "This is a test document about openBIS. It contains information about experiments and data management."
                )

            # Create test file with datastore source
            test_file2 = input_dir / "test2.txt"
            with open(test_file2, "w", encoding="utf-8") as f:
                f.write("Title: Test Document 2\n")
                f.write("URL: https://example.com/test2\n")
                f.write("Source: datastore\n")
                f.write("---\n\n")
                f.write(
                    "This is a test document about the data store. It explains how to upload and manage data files."
                )

            # Create processor
            processor = MultiSourceRAGProcessor(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                min_chunk_size=50,
                max_chunk_size=200,
            )

            # Process files
            processor.process()

            # Check output
            chunks_file = output_dir / "chunks.json"
            assert chunks_file.exists(), "Chunks file not created"

            with open(chunks_file, encoding="utf-8") as f:
                chunks = json.load(f)

            assert len(chunks) > 0, "No chunks created"

            # Check that sources are preserved
            sources = set(chunk.get("source") for chunk in chunks)
            assert "openbis" in sources, "openBIS source not found"
            assert "datastore" in sources, "datastore source not found"

            logger.info(
                f"✓ Processor created {len(chunks)} chunks with sources: {sources}"
            )
            return True

        except Exception as e:
            logger.error(f"✗ Processor test failed: {e}")
            return False


def test_vector_database():
    """Test the vector database functionality."""
    logger.info("Testing vector database...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test chunks
            test_chunks = [
                {
                    "chunk_id": "test_1",
                    "title": "Test Document 1",
                    "url": "https://example.com/test1",
                    "source": "openbis",
                    "content": "This is about openBIS experiments and data management.",
                    "embedding": [0.1] * 1536,  # Dummy embedding
                },
                {
                    "chunk_id": "test_2",
                    "title": "Test Document 2",
                    "url": "https://example.com/test2",
                    "source": "datastore",
                    "content": "This is about data store upload and file management.",
                    "embedding": [0.2] * 1536,  # Dummy embedding
                },
            ]

            # Create vector database
            db_path = Path(temp_dir) / "test_vectordb"
            vector_db = DesiVectorDB(
                db_path=str(db_path), collection_name="test_collection"
            )

            # Add chunks
            vector_db.add_chunks(test_chunks)

            # Test search
            query_embedding = [0.15] * 1536  # Should be closer to first chunk
            results = vector_db.search(query_embedding, n_results=2)

            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            assert all("chunk_id" in result for result in results), (
                "Missing chunk_id in results"
            )
            assert all("source" in result for result in results), (
                "Missing source in results"
            )

            # Test stats
            stats = vector_db.get_collection_stats()
            assert stats["total_chunks"] == 2, (
                f"Expected 2 chunks, got {stats['total_chunks']}"
            )

            vector_db.close()

            logger.info("✓ Vector database functionality works")
            return True

        except Exception as e:
            logger.error(f"✗ Vector database test failed: {e}")
            return False


def test_query_engine():
    """Test the query engine (without Ollama)."""
    logger.info("Testing query engine...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test vector database
            db_path = Path(temp_dir) / "test_vectordb"
            vector_db = DesiVectorDB(
                db_path=str(db_path), collection_name="test_collection"
            )

            # Add test chunks
            test_chunks = [
                {
                    "chunk_id": "test_1",
                    "title": "openBIS Experiments",
                    "url": "https://example.com/test1",
                    "source": "openbis",
                    "content": "How to create experiments in openBIS system.",
                    "embedding": [0.1] * 1536,
                },
                {
                    "chunk_id": "test_2",
                    "title": "Data Store Upload",
                    "url": "https://example.com/test2",
                    "source": "datastore",
                    "content": "How to upload files to the data store.",
                    "embedding": [0.2] * 1536,
                },
            ]

            vector_db.add_chunks(test_chunks)
            vector_db.close()

            # Create query engine
            query_engine = DesiRAGQueryEngine(
                db_path=str(db_path), collection_name="test_collection"
            )

            # Test retrieval
            results = query_engine.retrieve_relevant_chunks("experiments", top_k=2)
            assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"

            # Test stats
            stats = query_engine.get_database_stats()
            assert stats["total_chunks"] == 2, (
                f"Expected 2 chunks, got {stats['total_chunks']}"
            )

            query_engine.close()

            logger.info("✓ Query engine functionality works")
            return True

        except Exception as e:
            logger.error(f"✗ Query engine test failed: {e}")
            return False


def run_all_tests():
    """Run all tests."""
    logger.info("Starting DeSi system tests...")

    tests = [
        ("ReadTheDocs Scraper", test_readthedocs_scraper),
        ("Wiki.js Scraper", test_wikijs_scraper),
        ("Multi-Source Processor", test_processor),
        ("Vector Database", test_vector_database),
        ("Query Engine", test_query_engine),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        logger.info("🎉 All tests passed! DS Helper system is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1


def main():
    """Main function."""
    setup_logging(logging.INFO)

    logger.info("DS Helper System Test Suite")
    logger.info(
        "This will test core functionality without requiring external services."
    )

    return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
