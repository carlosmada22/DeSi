"""
Master CLI for the Unified RAG Knowledge Base Builder

This script serves as the single entry point for the entire data processing pipeline.
It orchestrates the deletion of the old database and the execution of specialized
processors for DSWiki and openBIS to build a fresh, unified ChromaDB vector store.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# MODIFIED: Import the processor classes instead of the run functions
from .ds_processor import DsWikiProcessor
from .openbis_processor import OpenBisProcessor

# --- Configure Logging (No changes needed) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- DEFAULT PATH CONFIGURATION (No changes needed) ---
DEFAULT_DSWIKI_INPUT = "./data/raw/wikijs/daily"
DEFAULT_OPENBIS_INPUT = "./data/raw/openbis/improved"
DEFAULT_OUTPUT_DIR = "./data/processed"
DEFAULT_CHROMA_DIR = "./desi_vectordb"
# ------------------------------------


def delete_existing_database(chroma_dir: str):
    """
    Safely deletes the specified ChromaDB directory to ensure a fresh build.
    (No changes needed in this function)
    """
    if os.path.exists(chroma_dir):
        logger.info(f"Deleting existing vector database at: {chroma_dir}")
        try:
            shutil.rmtree(chroma_dir)
            logger.info("Database deleted successfully.")
        except OSError as e:
            logger.error(f"Error deleting directory {chroma_dir}: {e}")
            sys.exit(1)
    else:
        logger.info(f"No existing database found at {chroma_dir}. Starting fresh.")


def main():
    """Main CLI entry point for the complete RAG pipeline."""
    # --- Argument Parsing (No changes needed) ---
    parser = argparse.ArgumentParser(
        description="Unified RAG Knowledge Base Builder CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full pipeline with default paths (most common use case):
  python -m desi.processor.cli

  # Override the ChromaDB location for a test run:
  python -m desi.processor.cli --chroma-dir ./test_vectordb

  # Run the pipeline but append to the DB instead of deleting it:
  python -m desi.processor.cli --no-delete
        """,
    )
    parser.add_argument(
        "--dswiki-input",
        type=str,
        default=DEFAULT_DSWIKI_INPUT,
        help=f"Input directory for DSWiki files. Default: {DEFAULT_DSWIKI_INPUT}",
    )
    parser.add_argument(
        "--openbis-input",
        type=str,
        default=DEFAULT_OPENBIS_INPUT,
        help=f"Input directory for openBIS files. Default: {DEFAULT_OPENBIS_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent directory for processed outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"Directory for the ChromaDB vector store. Default: {DEFAULT_CHROMA_DIR}",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete the existing ChromaDB; append to it instead.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Path Validation (No changes needed) ---
    for path in [args.dswiki_input, args.openbis_input]:
        if not os.path.exists(path):
            logger.error(
                f"Input directory does not exist: {path}. Please create it or specify a different path."
            )
            sys.exit(1)

    # --- Pipeline Execution (MODIFIED SECTION) ---
    logger.info("🚀 Starting Unified Knowledge Base Build Process...")
    logger.info("=" * 60)

    if not args.no_delete:
        delete_existing_database(args.chroma_dir)
    else:
        logger.warning(
            f"Flag --no-delete is set. Appending to existing DB at {args.chroma_dir}"
        )

    dswiki_output_dir = os.path.join(args.output_dir, "dswiki")
    openbis_output_dir = os.path.join(args.output_dir, "openbis")
    os.makedirs(dswiki_output_dir, exist_ok=True)
    os.makedirs(openbis_output_dir, exist_ok=True)

    try:
        # Step 1: Instantiate and run the DSWiki processor
        logger.info("Initializing DSWiki Processor...")
        dswiki_processor = DsWikiProcessor(
            root_directory=args.dswiki_input,
            output_directory=dswiki_output_dir,
            chroma_persist_directory=args.chroma_dir,
        )
        dswiki_processor.process()

        # Step 2: Instantiate and run the openBIS processor
        logger.info("\nInitializing openBIS Processor...")
        openbis_processor = OpenBisProcessor(
            root_directory=args.openbis_input,
            output_directory=openbis_output_dir,
            chroma_persist_directory=args.chroma_dir,
        )
        openbis_processor.process()

        logger.info("\n" + "=" * 60)
        logger.info(
            "✅ All processors completed successfully. Knowledge base is rebuilt and up to date."
        )
        logger.info(f"Final vector store is located at: {args.chroma_dir}")

    except Exception as e:
        logger.error(
            f"A critical error occurred during the pipeline: {e}", exc_info=args.verbose
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
