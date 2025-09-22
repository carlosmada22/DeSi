#!/usr/bin/env python3
"""
Script to ingest processed chunks into the ChromaDB vector database.

This script reads the processed chunks from JSON files and loads them
into the ChromaDB vector database for efficient similarity search.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from desi.utils.vector_db import DesiVectorDB
from desi.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def load_chunks_from_json(json_file: Path) -> list:
    """
    Load chunks from a JSON file.

    Args:
        json_file: Path to the JSON file containing chunks

    Returns:
        List of chunk dictionaries
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {json_file}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks from {json_file}: {e}")
        return []


def main():
    """Main function to ingest chunks into vector database."""
    parser = argparse.ArgumentParser(
        description="Ingest processed chunks into ChromaDB vector database"
    )
    parser.add_argument(
        "--chunks-file", 
        required=True, 
        help="Path to the JSON file containing processed chunks"
    )
    parser.add_argument(
        "--db-path",
        default="desi_vectordb",
        help="Path to the ChromaDB database directory"
    )
    parser.add_argument(
        "--collection-name",
        default="desi_docs",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset the collection before ingesting (delete existing data)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Validate input file
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return 1

    try:
        # Initialize vector database
        logger.info(f"Initializing vector database at {args.db_path}")
        vector_db = DesiVectorDB(
            db_path=args.db_path,
            collection_name=args.collection_name
        )

        # Reset collection if requested
        if args.reset:
            logger.info("Resetting collection...")
            vector_db.reset_collection()

        # Load chunks from JSON file
        logger.info(f"Loading chunks from {chunks_file}")
        chunks = load_chunks_from_json(chunks_file)

        if not chunks:
            logger.error("No chunks loaded, exiting")
            return 1

        # Validate chunks have embeddings
        chunks_with_embeddings = []
        for chunk in chunks:
            if 'embedding' in chunk and chunk['embedding']:
                chunks_with_embeddings.append(chunk)
            else:
                logger.warning(f"Chunk {chunk.get('chunk_id', 'unknown')} has no embedding, skipping")

        if not chunks_with_embeddings:
            logger.error("No chunks with embeddings found")
            return 1

        logger.info(f"Found {len(chunks_with_embeddings)} chunks with embeddings")

        # Ingest chunks into vector database
        logger.info("Ingesting chunks into vector database...")
        if chunks_with_embeddings:
            vector_db.add_chunks(chunks_with_embeddings)
        else:
            logger.warning("No chunks with embeddings to ingest")

        # Print statistics
        stats = vector_db.get_collection_stats()
        logger.info("Ingestion completed successfully!")
        logger.info(f"Collection statistics: {stats}")

        # Close database connection
        vector_db.close()

        return 0

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
