#!/usr/bin/env python3
"""
Command-line interface for the DeSi query engine.
"""

import argparse
import logging
import sys

from .query import DesiRAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query DeSi using RAG with vector database."
    )
    parser.add_argument(
        "--db-path",
        default="desi_vectordb",
        help="Path to the ChromaDB database directory",
    )
    parser.add_argument(
        "--collection-name", default="desi_docs", help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--model", default="gpt-oss:20b", help="The Ollama model to use for chat"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="The number of chunks to retrieve"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--stats", action="store_true", help="Show database statistics and exit"
    )

    return parser.parse_args(args)


def run_with_args(args):
    """Run the query engine with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Create the query engine
        query_engine = DesiRAGQueryEngine(
            db_path=args.db_path, collection_name=args.collection_name, model=args.model
        )

        # Show stats if requested
        if args.stats:
            stats = query_engine.get_database_stats()
            print("\nDatabase Statistics:")
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Collection name: {stats.get('collection_name', 'unknown')}")
            print(f"Source distribution: {stats.get('source_distribution', {})}")
            query_engine.close()
            return 0

        print("DeSi - RAG Query Engine")
        print("Type 'quit' or 'exit' to stop, 'stats' to show database statistics")
        print("-" * 50)

        while True:
            try:
                # Get user input
                query = input("\nYour question: ").strip()

                if not query:
                    continue

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if query.lower() == "stats":
                    stats = query_engine.get_database_stats()
                    print("\nDatabase Statistics:")
                    print(f"Total chunks: {stats.get('total_chunks', 0)}")
                    print(f"Collection name: {stats.get('collection_name', 'unknown')}")
                    print(
                        f"Source distribution: {stats.get('source_distribution', {})}"
                    )
                    continue

                # Process the query
                print("\nProcessing your question...")
                answer, relevant_chunks = query_engine.query(query, top_k=args.top_k)

                # Display the answer
                print("\nAnswer:")
                print(answer)

                # Display relevant chunks if verbose
                if args.verbose and relevant_chunks:
                    print(f"\nRelevant chunks used ({len(relevant_chunks)}):")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        source = chunk.get("source", "unknown")
                        title = chunk.get("title", "Unknown")
                        similarity = chunk.get("similarity_score", 0)
                        print(
                            f"{i}. [{source.upper()}] {title} (similarity: {similarity:.3f})"
                        )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")

        # Close the query engine
        query_engine.close()
        return 0

    except Exception as e:
        logger.error(f"Error initializing query engine: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
