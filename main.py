#!/usr/bin/env python3
"""
Main entry point for DeSi.

This script provides a complete workflow that can:
1. Check if ChromaDB database exists
2. If not, check if scraped data exists, if not run scrapers
3. If scraped data exists but no database, run processors
4. Start the query engine (CLI mode)
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules that will be used in functions
from langchain_community.chat_models import ChatOllama

from desi.processor import DsWikiProcessor, OpenBisProcessor
from desi.query.conversation_engine import ChatbotEngine, SqliteConversationMemory
from desi.query.query import RAGQueryEngine
from desi.scraper import OpenbisScraper
from desi.utils.config import DesiConfig
from desi.utils.logging import setup_logging

# from desi.web.app import create_app  # Disabled for now

# Configure logging
logger = setup_logging()


def check_chromadb_exists(config: DesiConfig) -> bool:
    """Check if ChromaDB database already exists."""
    project_root = Path(__file__).parent
    chroma_dir = project_root / config.db_path

    # Check if ChromaDB directory exists and has the required files
    return (
        chroma_dir.exists()
        and (chroma_dir / "chroma.sqlite3").exists()
        and len(list(chroma_dir.glob("*"))) > 1  # More than just the sqlite file
    )


def check_scraped_data(config: DesiConfig) -> bool:
    """Check if scraped data already exists."""
    project_root = Path(__file__).parent
    data_dir = project_root / config.data_dir / "raw"

    openbis_files = (
        list((data_dir / "openbis").glob("*.txt"))
        if (data_dir / "openbis").exists()
        else []
    )
    wikijs_files = (
        list((data_dir / "wikijs").glob("*.txt"))
        if (data_dir / "wikijs").exists()
        else []
    )

    return len(openbis_files) > 0 or len(wikijs_files) > 0


def run_scrapers(config: DesiConfig) -> bool:
    """Run the scrapers to collect data."""
    logger.info("🕷️  Running scrapers...")
    try:
        openbis_scraper = OpenbisScraper(
            base_url=config.openbis_url,
            output_dir=f"{config.data_dir}/raw/openbis",
        )
        openbis_scraper.scrape()
        logger.info("Scraping complete")
        return True
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return False


def run_processor(config: DesiConfig) -> bool:
    """Run the processor to create embeddings and vector database."""
    logger.info("⚙️  Running processor...")
    try:
        # Instantiate and run the processors directly
        dswiki_processor = DsWikiProcessor(
            root_directory=f"{config.data_dir}/raw/wikijs",
            output_directory=f"{config.processed_data_dir}/wikijs",
            chroma_persist_directory=config.db_path,
        )
        dswiki_processor.process()

        openbis_processor = OpenBisProcessor(
            root_directory=f"{config.data_dir}/raw/openbis",
            output_directory=f"{config.processed_data_dir}/openbis",
            chroma_persist_directory=config.db_path,
        )
        openbis_processor.process()

        logger.info("Processing complete")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def run_query_interface(config: DesiConfig) -> None:
    """Run the interactive query interface."""
    logger.info("💬 Starting interactive query interface...")
    try:
        project_root = Path(__file__).parent
        db_path = str(project_root / config.db_path)
        memory_db_path = str(project_root / config.memory_db_path)
        prompt_template_path = str(project_root / "prompts" / "desi_query_prompt.md")

        # Initialize the RAG engine
        rag_engine = RAGQueryEngine(
            chroma_persist_directory=db_path,
            prompt_template_path=prompt_template_path,
            embedding_model=config.embedding_model_name,
            llm_model=config.model_name,
        )

        # Initialize conversation memory
        memory = SqliteConversationMemory(
            db_path=memory_db_path,
            history_limit=config.history_limit,
        )

        # Initialize rewrite LLM
        rewrite_llm = ChatOllama(model=config.model_name)

        # Initialize the chatbot engine
        engine = ChatbotEngine(
            rag_engine=rag_engine,
            memory=memory,
            rewrite_llm=rewrite_llm,
        )

        print("\n" + "=" * 60)
        print("🤖 DeSi - Your openBIS and DataStore Assistant")
        print("=" * 60)
        print("Type 'exit' to end the conversation.")
        print("=" * 60 + "\n")

        # Use the existing chat session method
        engine.start_chat_session()

    except Exception as e:
        logger.error(f"Failed to start query interface: {e}")
        sys.exit(1)


def run_web_interface(config: DesiConfig) -> None:
    """Run the web interface (placeholder - not implemented yet)."""
    logger.info("🌐 Web interface not implemented yet, starting CLI interface...")
    run_query_interface(config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DeSi - DataStore Helper: RAG-focused chatbot for openBIS and DataStore documentation"
    )
    parser.add_argument(
        "--web", action="store_true", help="Start web interface instead of CLI"
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip scraping even if no data exists",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip processing and go directly to query interface",
    )
    parser.add_argument(
        "--force-scraping",
        action="store_true",
        help="Force scraping even if data already exists",
    )
    parser.add_argument(
        "--force-processing",
        action="store_true",
        help="Force processing even if processed data exists",
    )
    parser.add_argument("--config", help="Path to configuration file (.env)")

    args = parser.parse_args()

    # Load configuration
    config = DesiConfig(args.config)

    logger.info("🚀 Starting DeSi pipeline...")
    logger.info(f"Configuration: {config.to_dict()}")

    # Step 1: Check if ChromaDB database exists
    has_chromadb = check_chromadb_exists(config)

    if has_chromadb and not args.force_processing and not args.force_scraping:
        logger.info("✅ ChromaDB database already exists, starting query interface...")
    else:
        if not has_chromadb:
            logger.info("📊 No ChromaDB database found, need to process data...")

        # Step 2: Check and run scrapers if needed
        if not args.skip_scraping:
            has_data = check_scraped_data(config)

            if not has_data or args.force_scraping:
                if not has_data:
                    logger.info("📂 No scraped data found, running scrapers...")
                else:
                    logger.info("🔄 Force scraping requested...")

                if not run_scrapers(config):
                    logger.error("❌ Scraping failed!")
                    sys.exit(1)
            else:
                logger.info("✅ Scraped data already exists, skipping scraping")
        else:
            logger.info("⏭️  Skipping scraping as requested")

        # Step 3: Check and run processor if needed
        if not args.skip_processing:
            if not has_chromadb or args.force_processing:
                if not has_chromadb:
                    logger.info("📊 No ChromaDB database found, running processor...")
                else:
                    logger.info("🔄 Force processing requested...")

                if not run_processor(config):
                    logger.error("❌ Processing failed!")
                    sys.exit(1)
            else:
                logger.info("✅ ChromaDB database already exists, skipping processing")
        else:
            logger.info("⏭️  Skipping processing as requested")

    # Step 4: Start the interface
    if args.web:
        logger.info("🌐 Web interface not implemented yet, starting CLI interface...")
        run_query_interface(config)
    else:
        run_query_interface(config)


if __name__ == "__main__":
    main()
