#!/usr/bin/env python3
"""
Unified Processor for Multi-Source Documentation

This module orchestrates the entire processing pipeline for documentation
from multiple sources, combining normalization, chunking, and metadata handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .content_normalizer import ContentNormalizer
from .enhanced_chunker import EnhancedChunker
from .metadata_handler import MetadataHandler

# Try to import embedding functionality
try:
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_AVAILABLE = True

    # Test Ollama connection
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            OLLAMA_AVAILABLE = False
    except Exception:
        OLLAMA_AVAILABLE = False

except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import ChromaDB functionality
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import EmbeddingFunction

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

    class EmbeddingFunction:  # type: ignore[misc]
        """Fallback embedding interface when chromadb is unavailable."""

        def __call__(self, input):  # type: ignore[override]
            raise NotImplementedError(
                "EmbeddingFunction requires chromadb to be installed."
            )


logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using Ollama (adapted from existing processor.py)"""

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        if OLLAMA_AVAILABLE:
            try:
                self.embeddings_model = OllamaEmbeddings(model=model_name)
                # Test connection
                test_embedding = self.embeddings_model.embed_query("test")
                if not test_embedding:
                    raise RuntimeError("Ollama returned empty embedding")
                self.available = True
            except Exception as e:
                logger.warning(f"Ollama embedding function not available: {e}")
                self.available = False
        else:
            self.available = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        if not self.available or not input:
            # Return dummy embeddings if Ollama not available
            return [[0.0] * 768 for _ in input]  # nomic-embed-text dimension

        embeddings = []
        for text in input:
            try:
                embedding = self.embeddings_model.embed_query(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)  # Fallback

        return embeddings

    def name(self) -> str:
        return f"ollama-{self.model_name}"


def _sanitize_metadata(d: dict) -> dict:
    """
    Sanitize metadata for ChromaDB (only primitive values allowed).
    """
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _make_unique_id(rid: str, item: dict, seen: set, counter: int) -> str:
    """
    Ensure IDs are unique for ChromaDB.
    """
    if rid not in seen:
        return rid
    path = item.get("file_path", "")
    src = item.get("source_type", "")
    sec = item.get("section_title", "")
    base = f"{rid}|{path}|{src}|{sec}"
    suffix = abs(hash(base)) % (10**8)
    cand = f"{rid}::{suffix}"
    while cand in seen:
        counter += 1
        cand = f"{rid}::{suffix}-{counter}"
    return cand


class UnifiedProcessor:
    """Unified processor for multi-source documentation."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50,
        generate_embeddings: bool = True,
        chroma_dir: Optional[str] = None,
        collection_name: str = "docs",
    ):
        """
        Initialize the unified processor.

        Args:
            input_dir: Directory containing scraped files
            output_dir: Directory to save processed output
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            generate_embeddings: Whether to generate embeddings
            chroma_dir: Directory for ChromaDB storage (optional)
            collection_name: Name of ChromaDB collection
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.generate_embeddings = generate_embeddings and OLLAMA_AVAILABLE
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.use_chromadb = chroma_dir is not None and CHROMADB_AVAILABLE

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create ChromaDB directory if specified
        if self.chroma_dir:
            Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.normalizer = ContentNormalizer()
        self.chunker = EnhancedChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.metadata_handler = MetadataHandler()

        # Initialize embeddings if available
        if self.generate_embeddings:
            try:
                self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
                logger.info("Using Ollama for embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama embeddings: {e}")
                self.generate_embeddings = False
        else:
            logger.info("Embeddings disabled or Ollama not available")

    def process_all_files(self) -> List[Dict]:
        """
        Process all files in the input directory.

        Returns:
            List of processed chunk dictionaries
        """
        all_chunks = []

        # Find all text files recursively
        text_files = list(self.input_dir.rglob("*.txt"))
        logger.info(f"Found {len(text_files)} text files to process")

        for file_path in text_files:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Total processed chunks: {len(all_chunks)}")
        return all_chunks

    def process_file(self, file_path: Path) -> List[Dict]:
        """
        Process a single file through the complete pipeline.

        Args:
            file_path: Path to the file to process

        Returns:
            List of processed chunk dictionaries
        """
        logger.debug(f"Processing {file_path}")

        # Step 1: Normalize content
        normalized_data = self.normalizer.normalize_file(file_path)

        # Step 2: Extract comprehensive metadata
        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()

        base_metadata = self.metadata_handler.extract_metadata(file_path, raw_content)

        # Step 3: Chunk the normalized content
        chunks = self.chunker.chunk_content(
            normalized_data["content"], normalized_data["source"]
        )

        if not chunks:
            logger.warning(f"No chunks generated for {file_path}")
            return []

        # Step 4: Generate embeddings if enabled
        embeddings = []
        if self.generate_embeddings:
            try:
                embeddings = self._generate_embeddings(chunks)
            except Exception as e:
                logger.error(f"Error generating embeddings for {file_path}: {e}")
                embeddings = [None] * len(chunks)
        else:
            embeddings = [None] * len(chunks)

        # Step 5: Create final chunk records
        processed_chunks = []
        sections = self.chunker.split_by_headings(normalized_data["content"])

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Find the section this chunk belongs to
            section_title = self._find_section_for_chunk(chunk, sections)

            # Enrich metadata for this chunk
            chunk_metadata = self.metadata_handler.enrich_chunk_metadata(
                base_metadata, chunk, i, section_title
            )

            # Create chunk record
            chunk_record = {
                "id": f"docs:{chunk_metadata.get('source', 'unknown')}:{chunk_metadata.get('file_stem', 'unknown')}:{i}",
                "title": chunk_metadata.get("title", ""),
                "source": self.metadata_handler.create_source_url(chunk_metadata),
                "source_type": chunk_metadata.get("source", ""),
                "content": chunk,
                "section_title": section_title,
                "chunk_index": i,
                "chunk_id": chunk_metadata.get("chunk_id", ""),
                "content_type": chunk_metadata.get("content_type", "general"),
                "file_path": str(file_path),
                "processed_at": chunk_metadata.get("processed_at", ""),
                "source_priority": chunk_metadata.get("source_priority", 2),
                "metadata": chunk_metadata,
            }

            # Add embedding if available
            if embedding is not None:
                chunk_record["embedding"] = embedding

            processed_chunks.append(chunk_record)

        return processed_chunks

    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        if not self.generate_embeddings or not chunks:
            return [None] * len(chunks)

        try:
            # Use the batch embedding method
            embeddings = self.embeddings_model.embed_documents(chunks)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fallback to one-by-one if batch fails, or just return Nones
            return [None] * len(chunks)

    def _find_section_for_chunk(self, chunk: str, sections: List[tuple]) -> str:
        """Find which section a chunk belongs to."""
        chunk_lines = chunk.split("\n")

        # Look for the first heading in the chunk
        for line in chunk_lines:
            if line.strip().startswith("#"):
                heading = line.strip().lstrip("#").strip()
                return heading

        # If no heading found in chunk, try to match with sections
        for section_title, section_content in sections:
            if chunk[:100] in section_content:
                return section_title

        return "Introduction"

    def build_chromadb_collection(self, chunks: List[Dict]) -> None:
        """
        Build ChromaDB collection from processed chunks.

        Args:
            chunks: List of processed chunk dictionaries
        """
        if not self.use_chromadb:
            logger.warning("ChromaDB not available or not configured")
            return

        logger.info(f"Building ChromaDB collection '{self.collection_name}'...")

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=self.chroma_dir, settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding function
        embed_fn = OllamaEmbeddingFunction()

        # (re)create collection cleanly
        existing = {c.name for c in chroma_client.list_collections()}
        if self.collection_name in existing:
            logger.info(f"Deleting existing collection '{self.collection_name}'...")
            chroma_client.delete_collection(self.collection_name)

        collection = chroma_client.create_collection(
            name=self.collection_name, embedding_function=embed_fn
        )

        ids, documents, metadatas = [], [], []
        seen_ids = set()
        count = 0

        for item in chunks:
            text = item.get("content")
            if not text or not str(text).strip():
                continue

            raw_id = str(item.get("id", f"chunk_{count}"))
            rid = _make_unique_id(raw_id, item, seen_ids, count)
            seen_ids.add(rid)

            # Prepare metadata (compatible with server.py structure)
            meta_raw = {
                "source": item.get("source"),
                "title": item.get("title"),
                "section": item.get("section_title"),
                "timestamp": item.get("processed_at"),
                "repo": item.get("source_type"),
                "path": item.get("file_path"),
                "content_type": item.get("content_type"),
                "source_priority": item.get("source_priority"),
            }
            meta = _sanitize_metadata(meta_raw)

            ids.append(rid)
            documents.append(str(text))
            metadatas.append(meta)
            count += 1

        if not ids:
            raise RuntimeError("No valid records found in processed chunks")

        logger.info(f"Adding {count} chunks to ChromaDB...")
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(
            f"âœ… Successfully indexed {count} chunks in ChromaDB collection '{self.collection_name}'."
        )

    def save_processed_data(self, chunks: List[Dict], format: str = "both") -> None:
        """
        Save processed data in specified format(s).

        Args:
            chunks: List of processed chunk dictionaries
            format: Output format ('json', 'csv', 'jsonl', or 'both')
        """
        if not chunks:
            logger.warning("No chunks to save")
            return

        # Save as JSON
        if format in ["json", "both"]:
            json_file = self.output_dir / "enhanced_chunks.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(chunks)} chunks to {json_file}")

        # Save as CSV (without embeddings for readability)
        if format in ["csv", "both"]:
            csv_data = []
            for chunk in chunks:
                csv_row = {
                    k: v
                    for k, v in chunk.items()
                    if k != "embedding" and k != "metadata"
                }
                csv_data.append(csv_row)

            df = pd.DataFrame(csv_data)
            csv_file = self.output_dir / "enhanced_chunks.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved chunk metadata to {csv_file}")

        # Save as JSONL (compatible with all_repos_chunking.py format)
        if format in ["jsonl", "both"]:
            jsonl_file = self.output_dir / "enhanced_chunks.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    # Convert to JSONL format similar to all_repos_chunking.py
                    jsonl_record = {
                        "id": chunk["id"],
                        "source": chunk["source"],
                        "repo": chunk["source_type"],
                        "title": chunk["title"],
                        "section": chunk["section_title"],
                        "text": chunk["content"],
                        "timestamp": chunk["processed_at"],
                        "source_priority": chunk["source_priority"],
                        "content_type": chunk["content_type"],
                    }
                    f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
            logger.info(f"Saved chunks in JSONL format to {jsonl_file}")

    def process(
        self, output_format: str = "both", build_chromadb: bool = True
    ) -> Dict[str, int]:
        """
        Run the complete processing pipeline.

        Args:
            output_format: Output format ('json', 'csv', 'jsonl', or 'both')
            build_chromadb: Whether to build ChromaDB collection

        Returns:
            Processing statistics
        """
        logger.info(f"Starting processing of files in {self.input_dir}")

        # Step 1: Process all files
        chunks = self.process_all_files()

        # Step 2: Save processed data
        self.save_processed_data(chunks, output_format)

        # Step 3: Build ChromaDB collection if requested and configured
        chromadb_built = False
        if build_chromadb and self.use_chromadb:
            try:
                self.build_chromadb_collection(chunks)
                chromadb_built = True
            except Exception as e:
                logger.error(f"Failed to build ChromaDB collection: {e}")
        elif build_chromadb and not self.use_chromadb:
            logger.warning(
                "ChromaDB build requested but not configured (chroma_dir not provided)"
            )

        # Generate statistics
        stats = {
            "total_chunks": len(chunks),
            "files_processed": len(set(chunk["file_path"] for chunk in chunks)),
            "sources": len(set(chunk["source_type"] for chunk in chunks)),
            "embeddings_generated": sum(1 for chunk in chunks if "embedding" in chunk),
            "chromadb_built": chromadb_built,
        }

        logger.info(f"Processing complete: {stats}")
        return stats
