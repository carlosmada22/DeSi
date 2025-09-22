#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner

This script runs the complete end-to-end pipeline:
1. Process and chunk documents from both sources
2. Generate embeddings using Ollama
3. Store everything in ChromaDB for RAG queries

Usage:
    python scripts/run_complete_pipeline.py

Requirements:
    - Ollama server running with nomic-embed-text model
    - ChromaDB installed
    - Raw scraped files in data/raw/
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🚀 Starting Complete RAG Pipeline")
    print("=" * 50)
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Check if input directory exists
    input_dir = Path("data/raw")
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Please make sure you have scraped files in data/raw/")
        sys.exit(1)
    
    # Check for files
    openbis_files = list((input_dir / "openbis").glob("*.txt")) if (input_dir / "openbis").exists() else []
    wikijs_files = list((input_dir / "wikijs").glob("*.txt")) if (input_dir / "wikijs").exists() else []
    
    print(f"📁 Found files:")
    print(f"   OpenBIS: {len(openbis_files)} files")
    print(f"   Wiki.js: {len(wikijs_files)} files")
    
    if not openbis_files and not wikijs_files:
        print("❌ No files found to process!")
        sys.exit(1)
    
    print("🔄 Running complete pipeline...")
    print()

    try:
        # Import and run the processor directly
        from desi.processor.unified_processor import UnifiedProcessor

        processor = UnifiedProcessor(
            input_dir='data/raw',
            output_dir='data/processed',
            chroma_dir='./chroma_store',
            collection_name='docs'
        )

        stats = processor.process(output_format='jsonl', build_chromadb=True)
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print("📊 Final Statistics:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Sources: {stats['sources']}")
        print(f"   Embeddings generated: {stats['embeddings_generated']}")
        print(f"   ChromaDB built: {stats['chromadb_built']}")
        print()
        print("📁 Output files:")
        print("   ✅ Processed chunks: data/processed/enhanced_chunks.jsonl")
        print("   ✅ ChromaDB collection: ./chroma_store/")
        print("   ✅ Collection name: 'docs'")
        print()
        print("🔍 Next steps:")
        print("   1. Your ChromaDB collection is ready for RAG queries")
        print("   2. You can now use the collection in your server.py")
        print("   3. Update JSONL_PATH in server.py to point to the new file")
        print("   4. Start your FastAPI server for RAG queries")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
