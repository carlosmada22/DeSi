#!/usr/bin/env python3
"""
Main entry point for DeSi.

This script provides a complete workflow that can:
1. Run scrapers (if data doesn't exist)
2. Process the scraped data
3. Start the query engine or web interface
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules that will be used in functions
from desi.__main__ import main as scraper_main
from desi.processor.unified_processor import UnifiedProcessor
from desi.query.conversation_engine import DesiConversationEngine
from desi.utils.config import DesiConfig
from desi.utils.logging import setup_logging
from desi.web.app import create_app

# Configure logging
logger = setup_logging()


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
        result = scraper_main()
        return result == 0
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return False


def run_processor(config: DesiConfig) -> bool:
    """Run the processor to create embeddings and vector database."""
    logger.info("⚙️  Running processor...")
    try:
        project_root = Path(__file__).parent
        input_dir = project_root / config.data_dir / "raw"
        output_dir = project_root / config.processed_data_dir
        chroma_dir = project_root / config.db_path

        processor = UnifiedProcessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            min_chunk_size=config.min_chunk_size,
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
            generate_embeddings=True,
            chroma_dir=str(chroma_dir),
            collection_name=config.collection_name,
        )

        stats = processor.process(output_format="both", build_chromadb=True)
        logger.info(f"Processing complete: {stats}")
        return stats["total_chunks"] > 0

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

        engine = DesiConversationEngine(
            db_path=db_path,
            collection_name=config.collection_name,
            model=config.model_name,
            memory_db_path=memory_db_path,
            retrieval_top_k=config.retrieval_top_k,
            history_limit=config.history_limit,
        )

        print("\n" + "=" * 60)
        print("🤖 DeSi - Your openBIS and DataStore Assistant")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.")
        print("=" * 60 + "\n")

        session_id = None

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  help     - Show this help message")
                    print("  stats    - Show database statistics")
                    print("  clear    - Clear conversation memory")
                    print("  new      - Start a new conversation session")
                    print("  quit/exit/bye - Exit the program")
                    print()
                    continue

                if user_input.lower() == "stats":
                    stats = engine.get_database_stats()
                    print("\n📊 Database Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    print()
                    continue

                if user_input.lower() == "clear":
                    if session_id:
                        engine.clear_session_memory(session_id)
                        print("🧹 Conversation memory cleared.")
                    else:
                        print("No active session to clear.")
                    print()
                    continue

                if user_input.lower() == "new":
                    session_id = None
                    print("🆕 Starting new conversation session.")
                    print()
                    continue

                # Process the query
                response, session_id, metadata = engine.chat(user_input, session_id)

                print(f"\nDeSi: {response}")

                # Show metadata if verbose
                if metadata.get("rag_chunks_used", 0) > 0:
                    print(
                        f"\n📚 Used {metadata['rag_chunks_used']} documentation chunks"
                    )
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in query interface: {e}")
                print(f"❌ Error: {e}")

        engine.close()

    except Exception as e:
        logger.error(f"Failed to start query interface: {e}")
        sys.exit(1)


def run_web_interface(config: DesiConfig) -> None:
    """Run the web interface."""
    logger.info("🌐 Starting web interface...")
    try:
        project_root = Path(__file__).parent
        db_path = str(project_root / config.db_path)

        app = create_app(
            db_path=db_path,
            collection_name=config.collection_name,
            model=config.model_name,
        )

        # Update Flask configuration
        app.config["SECRET_KEY"] = config.secret_key

        print(
            f"\n🌐 Starting DeSi web interface at http://{config.web_host}:{config.web_port}"
        )
        print("Press Ctrl+C to stop the server")

        app.run(host=config.web_host, port=config.web_port, debug=config.web_debug)

    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        sys.exit(1)


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

    # Step 1: Check and run scrapers if needed
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

    # Step 2: Check and run processor if needed
    if not args.skip_processing:
        project_root = Path(__file__).parent
        chroma_dir = project_root / config.db_path
        processed_dir = project_root / config.processed_data_dir

        has_processed_data = (
            chroma_dir.exists()
            and (chroma_dir / "chroma.sqlite3").exists()
            and processed_dir.exists()
            and len(list(processed_dir.glob("*.json"))) > 0
        )

        if not has_processed_data or args.force_processing:
            if not has_processed_data:
                logger.info("📊 No processed data found, running processor...")
            else:
                logger.info("🔄 Force processing requested...")

            if not run_processor(config):
                logger.error("❌ Processing failed!")
                sys.exit(1)
        else:
            logger.info("✅ Processed data already exists, skipping processing")
    else:
        logger.info("⏭️  Skipping processing as requested")

    # Step 3: Start the interface
    if args.web:
        run_web_interface(config)
    else:
        run_query_interface(config)


if __name__ == "__main__":
    main()
