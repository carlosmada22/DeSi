#!/usr/bin/env python3
"""
CLI for Enhanced Multi-Source Documentation Processor

Complete end-to-end pipeline for processing documentation from multiple sources
(OpenBIS, Wiki.js) with intelligent chunking, embedding generation, and ChromaDB storage.
"""

import argparse
import logging
import sys
from pathlib import Path

from .unified_processor import UnifiedProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point for complete RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Source Documentation Processor with ChromaDB Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline: process, embed, and store in ChromaDB
  python -m desi.processor.cli --input-dir data/raw --output-dir data/processed

  # Process only (no ChromaDB)
  python -m desi.processor.cli --input-dir data/raw --output-dir data/processed --no-chromadb

  # Custom chunk sizes and ChromaDB directory
  python -m desi.processor.cli --input-dir data/raw --output-dir data/processed --chroma-dir ./my_chroma --min-chunk-size 150
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing scraped text files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save processed output'
    )

    # ChromaDB arguments
    parser.add_argument(
        '--chroma-dir',
        type=str,
        help='ChromaDB storage directory (enables ChromaDB if provided)'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default='docs',
        help='ChromaDB collection name (default: docs)'
    )
    
    # Chunking parameters
    parser.add_argument(
        '--min-chunk-size',
        type=int,
        default=100,
        help='Minimum chunk size in characters (default: 100)'
    )
    
    parser.add_argument(
        '--max-chunk-size',
        type=int,
        default=1000,
        help='Maximum chunk size in characters (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks in characters (default: 50)'
    )
    
    # Output format
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'jsonl', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    
    # Embeddings
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Disable embedding generation'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize and run processor
        logger.info("🚀 Starting Enhanced Multi-Source Processing...")
        logger.info("=" * 50)

        processor = UnifiedProcessor(
            input_dir=str(input_path),
            output_dir=str(output_path),
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            generate_embeddings=not args.no_embeddings,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name
        )

        # Process files
        build_chromadb = args.chroma_dir is not None
        stats = processor.process(output_format=args.format, build_chromadb=build_chromadb)

        # Print results
        logger.info("✅ Processing completed!")
        logger.info("=" * 50)
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Sources: {stats.get('sources', 'N/A')}")
        logger.info(f"Embeddings generated: {stats.get('embeddings_generated', 0)}")
        if build_chromadb:
            logger.info(f"ChromaDB built: {stats.get('chromadb_built', False)}")

        # Print output files
        output_files = list(output_path.glob("enhanced_chunks.*"))
        if output_files:
            logger.info("Output files:")
            for file in output_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  {file.name} ({size_mb:.1f} MB)")

        logger.info(f"Output directory: {output_path}")

        if build_chromadb and stats.get('chromadb_built'):
            logger.info(f"ChromaDB collection '{args.collection_name}' created in: {args.chroma_dir}")
            logger.info("Ready for RAG queries!")

        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
