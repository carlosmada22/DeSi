#!/usr/bin/env python3
"""
RAG Processor for ReadtheDocs Markdown Content

This script processes scraped Markdown files. It chunks the content using a
header-aware strategy, generates embeddings (using Ollama or dummy data),
and saves the output to JSON and CSV files for a RAG pipeline.
"""

import argparse
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- OLLAMA EMBEDDING SETUP ---
# Try to import Ollama embeddings, but don't fail if it's not available
try:
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_AVAILABLE = True
    try:
        # Check if Ollama server is running with a quick test
        logger.info("Checking for Ollama server...")
        embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        embeddings_model.embed_query("test")
        logger.info("Ollama server is running and embeddings are working.")
    except Exception as e:
        logger.warning(
            f"Could not connect to Ollama server. Will use dummy embeddings. Error: {e}"
        )
        OLLAMA_AVAILABLE = False
except ImportError:
    logger.warning("Langchain Ollama package not available. Using dummy embeddings.")
    OLLAMA_AVAILABLE = False


# --- YOUR INTELLIGENT CHUNKER CLASS (with minor cleanup) ---
class ContentChunker:
    """Class for chunking Markdown content into smaller, context-aware pieces."""

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        """
        Initialize the chunker.
        Note: The original 'chunk_overlap' was removed as it was not used in the logic.
        Context is maintained by prepending headers, which is a more robust method.

        Args:
            min_chunk_size: The minimum size of a chunk in characters.
            max_chunk_size: The maximum size of a chunk in characters.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_content(self, content: str) -> List[str]:
        """
        Chunk the Markdown content into smaller pieces based on headers.

        Args:
            content: The Markdown content to chunk.

        Returns:
            A list of content chunks.
        """
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        section_heading = None
        subsection_heading = None

        for paragraph in paragraphs:
            is_main_heading = paragraph.strip().startswith("# ")
            is_section_heading = paragraph.strip().startswith("## ")
            is_subsection_heading = paragraph.strip().startswith("### ")

            if is_main_heading or is_section_heading:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                section_heading = paragraph
                subsection_heading = None
                continue

            if is_subsection_heading:
                subsection_heading = paragraph
                if len(current_chunk) >= self.max_chunk_size * 0.7:
                    if current_chunk and len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())
                    current_chunk = (
                        (section_heading + "\n\n" + subsection_heading)
                        if section_heading
                        else subsection_heading
                    )
                else:
                    current_chunk += "\n\n" + subsection_heading
                continue

            if (
                len(current_chunk) + len(paragraph) > self.max_chunk_size
                and len(current_chunk) >= self.min_chunk_size
            ):
                chunks.append(current_chunk.strip())
                new_chunk_prefix = ""
                if section_heading:
                    new_chunk_prefix += section_heading + "\n\n"
                if subsection_heading and subsection_heading != section_heading:
                    new_chunk_prefix += subsection_heading + "\n\n"
                current_chunk = new_chunk_prefix

            current_chunk += paragraph + "\n\n"

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks


# --- YOUR EMBEDDING GENERATOR (unchanged) ---
class EmbeddingGenerator:
    """Class for generating embeddings for content chunks."""

    def __init__(self):
        if OLLAMA_AVAILABLE:
            logger.info("Using Ollama for embeddings with model 'nomic-embed-text'")
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            logger.warning("Using dummy embeddings (random vectors)")
            self.embeddings_model = None

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if OLLAMA_AVAILABLE and self.embeddings_model:
            try:
                return self.embeddings_model.embed_documents(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings with Ollama: {e}")
                logger.warning("Falling back to dummy embeddings.")
        return [self._generate_dummy_embedding() for _ in texts]

    def _generate_dummy_embedding(self, dim: int = 768) -> List[float]:
        # nomic-embed-text has a dimension of 768
        vec = np.random.normal(0, 1, dim)
        vec /= np.linalg.norm(vec)
        return vec.tolist()


# --- YOUR RAG PROCESSOR (modified for .md files and metadata reconstruction) ---
class RAGProcessor:
    """Class for processing content for RAG."""

    def __init__(
        self, input_dir: str, output_dir: str, min_chunk_size: int, max_chunk_size: int
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = ContentChunker(min_chunk_size, max_chunk_size)
        self.embedding_generator = EmbeddingGenerator()

    def _clean_markdown_content(self, content: str) -> str:
        """Cleans raw markdown content with robust, multi-stage logic."""
        # 1. Remove permalink artifacts
        content = re.sub(r'\[\]\(.*? "Permalink to this heading"\)', "", content)

        # 2. **ROBUST DEDENTING LOGIC**
        # Process paragraph by paragraph to find and fix indented code blocks
        paragraphs = content.split("\n\n")
        cleaned_paragraphs = []
        for p in paragraphs:
            # Check if the paragraph starts with whitespace but is not a list or header
            if p and p[0].isspace() and not p.strip().startswith(("*", "-", "#")):
                # textwrap.dedent intelligently removes common leading whitespace
                p = textwrap.dedent(p)
            cleaned_paragraphs.append(p)
        content = "\n\n".join(cleaned_paragraphs)

        # 3. Collapse excessive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _metadata_from_filename(self, file_path: Path) -> Dict[str, str]:
        """Reconstructs URL and Title from the scraped filename."""
        base_url = "https://openbis.readthedocs.io/"
        # en_20.10.0-11_user-documentation_general-users.md -> en/20.10.0-11/user-documentation/general-users.html
        parts = file_path.stem.split("_")
        url_path = "/".join(parts) + ".html"
        full_url = base_url + url_path

        # Create a simple title from the last part of the filename
        title = parts[-1].replace("-", " ").capitalize()
        return {"url": full_url, "title": title}

    def process_file(self, file_path: Path) -> List[Dict]:
        logger.info(f"Processing {file_path.name}")
        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()

        cleaned_content = self._clean_markdown_content(raw_content)
        metadata = self._metadata_from_filename(file_path)

        chunks = self.chunker.chunk_content(cleaned_content)
        if not chunks:
            logger.warning(f"No chunks generated for {file_path.name}")
            return []

        embeddings = self.embedding_generator.generate_embeddings(chunks)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append(
                {
                    "title": metadata["title"],
                    "url": metadata["url"],
                    "content": chunk,
                    "chunk_id": f"{file_path.stem}_{i}",
                }
            )
        return processed_chunks

    def process_all_files(self) -> List[Dict]:
        all_chunks = []
        # MODIFICATION: Look for .md files instead of .txt
        for file_path in sorted(self.input_dir.glob("*.md")):
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return all_chunks

    def save_processed_data(self, chunks: List[Dict]):
        if not chunks:
            logger.warning("No chunks were processed. Nothing to save.")
            return

        # Save the chunks as a standard JSON file
        chunks_file = self.output_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

        # Save the chunks as a JSON Lines file
        jsonl_file = self.output_dir / "chunks.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Convert each dictionary to a JSON string and write it on its own line
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {jsonl_file}")

        # Create a DataFrame for the CSV file (for easier viewing)
        df = pd.DataFrame(
            [{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks]
        )
        csv_file = self.output_dir / "chunks.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved chunk metadata for review to {csv_file}")

    def process(self):
        logger.info(f"Starting processing of files in {self.input_dir}")
        chunks = self.process_all_files()
        self.save_processed_data(chunks)
        logger.info(f"Finished processing. Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Markdown files for a RAG pipeline."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing the scraped .md files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the processed chunks (JSON and CSV).",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        help="Minimum size of a chunk in characters.",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1500,
        help="Maximum size of a chunk in characters.",
    )
    args = parser.parse_args()

    processor = RAGProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )
    processor.process()
