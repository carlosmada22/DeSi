#!/usr/bin/env python3
"""
Test ChromaDB Collection

This script tests the ChromaDB collection created by the pipeline
to ensure it's working correctly for RAG queries.

Usage:
    python tests/test_chromadb_collection.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ ChromaDB not installed. Install with: pip install chromadb")
    sys.exit(1)


def test_collection():
    """Test the ChromaDB collection."""

    chroma_dir = "./chroma_store"
    collection_name = "docs"

    # Check if ChromaDB directory exists
    if not Path(chroma_dir).exists():
        print(f"❌ ChromaDB directory not found: {chroma_dir}")
        print("Run the pipeline first: python scripts/run_complete_pipeline.py")
        return False

    try:
        # Initialize ChromaDB client
        print(f"🔗 Connecting to ChromaDB at: {chroma_dir}")
        client = chromadb.Client(
            Settings(persist_directory=chroma_dir, anonymized_telemetry=False)
        )

        # List collections
        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        print(f"📚 Available collections: {collection_names}")

        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' not found!")
            return False

        # Get the collection
        collection = client.get_collection(name=collection_name)

        # Get collection info
        count = collection.count()
        print(f"📊 Collection '{collection_name}' contains {count} documents")

        if count == 0:
            print("❌ Collection is empty!")
            return False

        # Test a simple query
        print("\n🔍 Testing sample queries...")

        test_queries = [
            "What is OpenBIS?",
            "How to create a dataset?",
            "Python API documentation",
            "user management",
        ]

        for query in test_queries:
            print(f"\n   Query: '{query}'")
            try:
                results = collection.query(query_texts=[query], n_results=3)

                if results["documents"] and results["documents"][0]:
                    print(f"   ✅ Found {len(results['documents'][0])} results")

                    # Show first result preview
                    first_doc = results["documents"][0][0]
                    first_meta = (
                        results["metadatas"][0][0] if results["metadatas"][0] else {}
                    )

                    print("   📄 Top result:")
                    print(f"      Title: {first_meta.get('title', 'N/A')}")
                    print(f"      Source: {first_meta.get('source', 'N/A')}")
                    print(f"      Preview: {first_doc[:100]}...")
                else:
                    print("   ❌ No results found")

            except Exception as e:
                print(f"   ❌ Query failed: {e}")
                return False

        print(f"\n✅ ChromaDB collection '{collection_name}' is working correctly!")
        print(f"   Ready for RAG queries with {count} documents")
        return True

    except Exception as e:
        print(f"❌ Error testing collection: {e}")
        return False


def main():
    print("🧪 Testing ChromaDB Collection")
    print("=" * 40)

    success = test_collection()

    if success:
        print("\n🎉 All tests passed!")
        print("Your ChromaDB collection is ready for RAG queries.")
    else:
        print("\n❌ Tests failed!")
        print("Check the error messages above and run the pipeline again if needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
