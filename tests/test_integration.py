import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

# Add the src directory to the path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the actual classes from your source code
from desi.processor.ds_processor import DsWikiProcessor
from desi.processor.openbis_processor import OpenBisProcessor
from desi.query.query import RAGQueryEngine
from desi.scraper.openbis_scraper import OpenbisScraper


# This marker is crucial for separating slow tests from fast ones
@pytest.mark.integration
def test_full_pipeline_integration(tmp_path: Path):
    """
    Tests the complete DeSi pipeline: scrape -> process -> query.
    This is a slow test that requires a running Ollama instance.

    Args:
        tmp_path (Path): A pytest fixture providing a temporary directory path.
    """
    # --- 1. SETUP: Define temporary directories for the test ---
    raw_data_dir = tmp_path / "raw"
    processed_data_dir = tmp_path / "processed"
    db_dir = tmp_path / "desi_vectordb"

    openbis_raw_dir = raw_data_dir / "openbis"
    wikijs_raw_dir = raw_data_dir / "wikijs"
    openbis_raw_dir.mkdir(parents=True)
    wikijs_raw_dir.mkdir(parents=True)

    # --- 2. SCRAPING: Scrape a few pages of the openBIS documentation ---
    print("\n--- Testing Scraping ---")

    # We will manually limit the scraper to 3 pages for a fast test.
    max_pages_to_scrape = 3
    scraper = OpenbisScraper(
        base_url="https://openbis.readthedocs.io/en/20.10.0-11/",
        output_dir=str(openbis_raw_dir),
    )

    # Manually run the scraping loop to control the number of pages
    scraped_count = 0
    while scraper.to_visit and scraped_count < max_pages_to_scrape:
        current_url = scraper.to_visit.pop()
        if current_url in scraper.visited:
            continue

        try:
            response = scraper._fetch_page(current_url)
            scraper.visited.add(current_url)
            soup = BeautifulSoup(response.content, "lxml")

            scraper._save_content_as_markdown(soup, current_url)
            scraper._find_new_links(soup, current_url)
            scraped_count += 1
            print(f"Scraped page {scraped_count}/{max_pages_to_scrape}: {current_url}")
        except Exception as e:
            print(f"Warning: Could not scrape {current_url}. Error: {e}")

    scraped_files = list(openbis_raw_dir.glob("*.md"))
    assert scraped_files, "Scraping step did not produce any files."
    print(f"Scraping successful, created {len(scraped_files)} files.")

    # --- 3. PROCESSING: Run both processors to create the unified database ---
    print("\n--- Testing Processing ---")

    # Create a dummy WikiJS markdown file for the DsWikiProcessor to find
    dummy_wikijs_file = wikijs_raw_dir / "test-page.md"
    dummy_wikijs_file.write_text("""---
title: Test WikiJS Page
---
## Introduction
This is a test page from the BAM Data Store. It contains important information about data handling.
""")

    # Instantiate and run the openBIS processor
    openbis_processor = OpenBisProcessor(
        root_directory=str(openbis_raw_dir),
        output_directory=str(processed_data_dir / "openbis"),
        chroma_persist_directory=str(db_dir),
    )
    openbis_processor.process()

    # Instantiate and run the DSWiki processor (which appends to the same DB)
    dswiki_processor = DsWikiProcessor(
        root_directory=str(wikijs_raw_dir),
        output_directory=str(processed_data_dir / "wikijs"),
        chroma_persist_directory=str(db_dir),
    )
    dswiki_processor.process()

    assert db_dir.exists(), "Vector database directory was not created."
    assert any(db_dir.iterdir()), "Vector database directory is empty after processing."
    print("Processing successful, vector database created.")

    # --- 4. QUERYING: Test the RAG engine against the created database ---
    print("\n--- Testing Querying ---")

    # Create a dummy prompt template file
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Context: {context_str}\n\nQuery: {query}\n\nAnswer:")

    # Instantiate the RAG engine
    rag_engine = RAGQueryEngine(
        chroma_persist_directory=str(db_dir), prompt_template_path=str(prompt_file)
    )

    # Ensure the engine loaded the vector store correctly
    assert rag_engine.vector_store is not None, (
        "RAG engine failed to load the vector store."
    )

    # Perform a query that should hit the scraped content
    test_query = "What is openBIS?"
    answer, source_chunks = rag_engine.query(test_query)

    assert isinstance(answer, str) and answer, (
        "RAG engine did not return a valid answer string."
    )
    assert source_chunks, "RAG engine did not return any source chunks for the query."

    # Check that a source from openBIS was retrieved
    origins = {chunk.metadata.get("origin") for chunk in source_chunks}
    assert "openbis" in origins, (
        "The retrieved chunks did not include a source from 'openbis'."
    )

    print(f"Query test successful. Received answer: '{answer[:100]}...'")
    print("✅ Integration test passed!")
