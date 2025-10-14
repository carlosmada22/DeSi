#!/usr/bin/env python3
"""
Main Orchestrator for Building the Unified RAG Knowledge Base

This script manages the complete, automated pipeline for building the vector store.
It imports and executes specialized processors for each data source in sequence.

Workflow:
1. Deletes the existing ChromaDB directory for a fresh build.
2. Runs the DSWiki processor to ingest its data.
3. Runs the openBIS processor to ingest its data into the same database.
"""

import logging
import os
import shutil
import sys

# Import the main processing functions from your other scripts
from ds_processor import run_dswiki_processing
from openbis_processor import run_openbis_processing

# PATHS
DSWIKI_INPUT_DIR = "./data/raw/wikijs/daily"
OPENBIS_INPUT_DIR = "./data/raw/openbis/improved"

# Parent directory for intermediate files (JSON, etc.)
# Subdirectories will be created automatically.
PROCESSED_DATA_DIR = "./data/processed"

# The final, unified vector database directory.
CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"
# ---------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def delete_existing_database(chroma_dir: str):
    """
    Safely deletes the specified ChromaDB directory to ensure a fresh build.
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


if __name__ == "__main__":
    logger.info("🚀 Starting Unified Knowledge Base Build Process...")

    # Step 1: Delete the old database for a clean slate.
    delete_existing_database(CHROMA_PERSIST_DIRECTORY)

    # Define specific output directories for each source
    dswiki_output_dir = os.path.join(PROCESSED_DATA_DIR, "dswiki")
    openbis_output_dir = os.path.join(PROCESSED_DATA_DIR, "openbis")
    os.makedirs(dswiki_output_dir, exist_ok=True)
    os.makedirs(openbis_output_dir, exist_ok=True)

    try:
        # Step 2: Run the DSWiki processor
        run_dswiki_processing(
            root_directory=DSWIKI_INPUT_DIR,
            output_directory=dswiki_output_dir,
            chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        # Step 3: Run the openBIS processor
        run_openbis_processing(
            root_directory=OPENBIS_INPUT_DIR,
            output_directory=openbis_output_dir,
            chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        logger.info(
            "\n✅ All processors completed successfully. Knowledge base is rebuilt and up to date."
        )

    except Exception as e:
        logger.error(f"An error occurred during the pipeline: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
