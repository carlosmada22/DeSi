#!/usr/bin/env python3
"""
Command-line interface for the DeSi web interface.
"""

import argparse
import logging
import sys

from .app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DeSi web interface.")
    parser.add_argument("--db-path", default="desi_vectordb", help="Path to the ChromaDB database directory")
    parser.add_argument("--collection-name", default="desi_docs", help="Name of the ChromaDB collection")
    parser.add_argument("--model", default="gpt-oss:20b", help="The Ollama model to use for chat")
    parser.add_argument("--host", default="127.0.0.1", help="The host to run the web interface on")
    parser.add_argument("--port", type=int, default=5000, help="The port to run the web interface on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the web interface with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Create the Flask app
        app = create_app(
            db_path=args.db_path,
            collection_name=args.collection_name,
            model=args.model
        )

        logger.info(f"Starting DeSi web interface on {args.host}:{args.port}")
        logger.info(f"Using database: {args.db_path}")
        logger.info(f"Using collection: {args.collection_name}")
        logger.info(f"Using model: {args.model}")

        # Run the Flask app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )

        return 0

    except Exception as e:
        logger.error(f"Error starting web interface: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
