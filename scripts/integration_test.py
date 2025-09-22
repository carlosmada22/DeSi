#!/usr/bin/env python3
"""
Integration test for DeSi.

This script tests the complete DeSi pipeline with a small sample
of real data to ensure everything works together correctly.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.scraper.cli import parse_args as scraper_parse_args, run_with_args as scraper_run
from desi.processor.cli import parse_args as processor_parse_args, run_with_args as processor_run
from desi.query.query import DesiRAGQueryEngine
from desi.utils.logging import setup_logging
from ingest_to_vectordb import main as ingest_main

# Configure logging
logger = logging.getLogger(__name__)


def test_scraping(temp_dir: Path, max_pages: int = 3):
    """Test scraping a small amount of real data."""
    logger.info("=== Testing Scraping ===")
    
    raw_dir = temp_dir / "raw"
    openbis_dir = raw_dir / "openbis"
    openbis_dir.mkdir(parents=True)
    
    try:
        # Test ReadTheDocs scraping
        logger.info("Testing ReadTheDocs scraping...")
        scraper_args = scraper_parse_args([
            "readthedocs",
            "--url", "https://openbis.readthedocs.io/en/20.10.0-11/",
            "--output", str(openbis_dir),
            "--max-pages", str(max_pages),
            "--verbose"
        ])
        
        result = scraper_run(scraper_args)
        if result != 0:
            logger.error("ReadTheDocs scraping failed")
            return False
        
        # Check if files were created
        scraped_files = list(openbis_dir.glob("*.txt"))
        if not scraped_files:
            logger.error("No files were scraped")
            return False
        
        logger.info(f"✓ Successfully scraped {len(scraped_files)} files")
        
        # Check file content
        sample_file = scraped_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.startswith("Title:"):
            logger.error("Scraped file doesn't have expected format")
            return False
        
        if "Source: openbis" not in content:
            logger.error("Source metadata not added correctly")
            return False
        
        logger.info("✓ Scraped files have correct format and metadata")
        return True
        
    except Exception as e:
        logger.error(f"Scraping test failed: {e}")
        return False


def test_processing(temp_dir: Path):
    """Test processing the scraped data."""
    logger.info("=== Testing Processing ===")
    
    raw_dir = temp_dir / "raw"
    processed_dir = temp_dir / "processed"
    
    try:
        # First, let's check what files we have and their content
        scraped_files = list(raw_dir.glob("**/*.txt"))
        logger.info(f"Found {len(scraped_files)} scraped files:")
        for f in scraped_files:
            logger.info(f"  - {f}")
            # Check file size and first few lines
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"    Size: {len(content)} characters")
                lines = content.split('\n')[:10]
                logger.info(f"    First 10 lines: {lines}")

        processor_args = processor_parse_args([
            "--input", str(raw_dir),
            "--output", str(processed_dir),
            "--min-chunk-size", "20",  # Very small for testing
            "--verbose"
        ])

        result = processor_run(processor_args)
        if result != 0:
            logger.error("Processing failed")
            return False
        
        # Check if output files were created
        chunks_file = processed_dir / "chunks.json"
        csv_file = processed_dir / "chunks.csv"
        
        if not chunks_file.exists():
            logger.error("chunks.json not created")
            return False
        
        if not csv_file.exists():
            logger.error("chunks.csv not created")
            return False
        
        # Check chunks content
        import json
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            logger.error("No chunks created")
            return False
        
        # Verify chunk structure
        sample_chunk = chunks[0]
        required_fields = ['title', 'url', 'source', 'content', 'embedding', 'chunk_id']
        for field in required_fields:
            if field not in sample_chunk:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check source metadata
        sources = set(chunk.get('source') for chunk in chunks)
        if 'openbis' not in sources:
            logger.error("openBIS source not found in chunks")
            return False
        
        logger.info(f"✓ Successfully processed {len(chunks)} chunks")
        logger.info(f"✓ Sources found: {sources}")
        return True
        
    except Exception as e:
        logger.error(f"Processing test failed: {e}")
        return False


def test_vector_database(temp_dir: Path):
    """Test vector database ingestion."""
    logger.info("=== Testing Vector Database Ingestion ===")
    
    processed_dir = temp_dir / "processed"
    db_dir = temp_dir / "vectordb"
    chunks_file = processed_dir / "chunks.json"
    
    try:
        # Save original argv and modify for ingestion
        original_argv = sys.argv.copy()
        sys.argv = [
            "ingest_to_vectordb",
            "--chunks-file", str(chunks_file),
            "--db-path", str(db_dir),
            "--reset",
            "--verbose"
        ]
        
        result = ingest_main()
        sys.argv = original_argv
        
        if result != 0:
            logger.error("Vector database ingestion failed")
            return False
        
        # Check if database was created
        if not db_dir.exists():
            logger.error("Vector database directory not created")
            return False
        
        logger.info("✓ Vector database ingestion completed")
        return True
        
    except Exception as e:
        logger.error(f"Vector database test failed: {e}")
        return False


def test_querying(temp_dir: Path):
    """Test querying the system."""
    logger.info("=== Testing Query Engine ===")
    
    db_dir = temp_dir / "vectordb"
    
    try:
        # Create query engine
        query_engine = DesiRAGQueryEngine(
            db_path=str(db_dir),
            collection_name="desi_docs"
        )
        
        # Test database stats
        stats = query_engine.get_database_stats()
        logger.info(f"Database stats: {stats}")
        
        if stats.get('total_chunks', 0) == 0:
            logger.error("No chunks found in database")
            return False
        
        # Test retrieval
        test_queries = [
            "openBIS",
            "experiment",
            "data management",
            "documentation"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = query_engine.retrieve_relevant_chunks(query, top_k=3)
            
            if not results:
                logger.warning(f"No results for query: '{query}'")
                continue
            
            logger.info(f"  Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                source = result.get('source', 'unknown')
                title = result.get('title', 'Unknown')
                similarity = result.get('similarity_score', 0)
                logger.info(f"  {i}. [{source.upper()}] {title} (similarity: {similarity:.3f})")
        
        query_engine.close()
        
        logger.info("✓ Query engine working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Query engine test failed: {e}")
        return False


def main():
    """Main function for integration test."""
    parser = argparse.ArgumentParser(description="Run DS Helper integration test")
    parser.add_argument("--max-pages", type=int, default=3, help="Maximum pages to scrape for testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger.info("DS Helper Integration Test")
    logger.info("This test runs the complete pipeline with real data")
    logger.info(f"Max pages to scrape: {args.max_pages}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Using temporary directory: {temp_path}")
        
        # Run tests in sequence
        tests = [
            ("Scraping", lambda: test_scraping(temp_path, args.max_pages)),
            ("Processing", lambda: test_processing(temp_path)),
            ("Vector Database", lambda: test_vector_database(temp_path)),
            ("Querying", lambda: test_querying(temp_path)),
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                results.append((test_name, result))
                if not result:
                    logger.error(f"{test_name} failed, stopping integration test")
                    break
            except Exception as e:
                logger.error(f"{test_name} failed with exception: {e}")
                results.append((test_name, False))
                break
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        
        passed = 0
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        if passed == len(tests):
            logger.info("🎉 Integration test passed! DS Helper is working correctly.")
            logger.info("\nYou can now run the full pipeline with:")
            logger.info("  python main.py")
            return 0
        else:
            logger.error("❌ Integration test failed. Please check the logs above.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
