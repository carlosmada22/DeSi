#!/usr/bin/env python3
"""
RAG Processor for ReadtheDocs Markdown Content

This script processes scraped Markdown files. It chunks the content using a
header-aware strategy, generates embeddings (using Ollama or dummy data),
and saves the output to JSON and CSV files for a RAG pipeline.
"""

import csv
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

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


# --- Helper Class for Storing Chunks (from ds_processor) ---
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        metadata_str = ", ".join(f"{k}='{v}'" for k, v in self.metadata.items())
        content_preview = self.page_content[:200].strip().replace("\n", " ")
        return f"Document(page_content='{content_preview}...', metadata={{{metadata_str}}})"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # This is included for compatibility with the ds_processor export format,
        # though this script doesn't handle datetime objects.
        return super().default(o)


class ContentChunker:
    """Class for chunking Markdown content into smaller, context-aware pieces."""

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        """
        Initialize the chunker.
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

        # If no chunks were created but there was content, return the whole content as one chunk.
        if not chunks and content:
            return [content.strip()]

        return chunks


class OpenBisProcessor:
    """
    Encapsulates the entire processing pipeline for the openBIS data source.
    """

    def __init__(self, root_directory, output_directory, chroma_persist_directory):
        self.root_directory = root_directory
        self.output_directory = output_directory
        self.chroma_persist_directory = chroma_persist_directory

    def process(self):
        """
        Executes the full processing pipeline. Replaces run_openbis_processing.
        """
        print("\n--- Starting openBIS Processing ---")

        # Step 1: Process all markdown files into chunks with enriched metadata
        final_chunks = self._process_all_openbis_files(self.root_directory)

        if final_chunks:
            # Step 2 (Optional): Export chunks to files for inspection
            self._export_chunks(final_chunks, self.output_directory)

            # Step 3: Generate embeddings and save them to ChromaDB
            self._create_and_persist_vectordb(
                final_chunks, self.chroma_persist_directory
            )

            print(
                f"--- openBIS Processing Complete. Total Chunks: {len(final_chunks)} ---"
            )
        else:
            print("No openBIS markdown files were found or processed.")

    @staticmethod
    def _clean_markdown_content(content: str) -> str:
        """Cleans raw markdown content with robust, multi-stage logic."""
        # 1. **THE KEY FIX**: Normalize whitespace. Replace the non-breaking space
        # character (U+00A0) with a regular space. This is the root cause of the issue.
        content = content.replace("\u00a0", " ")

        # 2. Remove permalink artifacts
        content = re.sub(r'\[\]\(.*? "Permalink to this heading"\)', "", content)

        # 3. Process paragraph by paragraph to find and fix indented code blocks
        paragraphs = content.split("\n\n")
        cleaned_paragraphs = []
        for p in paragraphs:
            # Check if the paragraph starts with whitespace but is not a list/header
            if p and p[0].isspace() and not p.strip().startswith(("*", "-", "#")):
                p = textwrap.dedent(p)
            cleaned_paragraphs.append(p)
        content = "\n\n".join(cleaned_paragraphs)

        # 4. Collapse excessive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _chunk_openbis_document(
        self, file_path: str, root_directory: str
    ) -> List[Document]:
        """
        Loads, cleans, enriches, and chunks a single openBIS markdown file.
        """
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, encoding="utf-8") as f:
                raw_content = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []

        # 1. Clean the specific ReadtheDocs artifacts
        cleaned_content = self._clean_markdown_content(raw_content)

        # 2. Enrich Metadata
        metadata = {}
        path_obj = Path(file_path)
        relative_path = os.path.relpath(file_path, root_directory)

        metadata["origin"] = "openbis"
        metadata["source"] = relative_path.replace("\\", "/")

        # Reconstruct original URL and create a title
        base_url = "https://openbis.readthedocs.io/"
        parts = path_obj.stem.split("_")
        metadata["url"] = base_url + "/".join(parts) + ".html"
        metadata["title"] = parts[-1].replace("-", " ").capitalize()

        # Create section from path structure (e.g., 'user-documentation')
        if len(parts) > 2:
            metadata["section"] = parts[2].replace("-", " ").title()
        else:
            metadata["section"] = "General"

        metadata["id"] = f"openbis-{path_obj.stem.lower()}"

        # 3. Chunk the document
        chunker = ContentChunker()
        content_chunks = chunker.chunk_content(cleaned_content)

        # 4. Create Document objects with enriched metadata
        final_chunks = []
        for i, content in enumerate(content_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["id"] += f"-{i}"  # Make chunk ID unique
            final_chunks.append(Document(page_content=content, metadata=chunk_metadata))

        return final_chunks

    def _process_all_openbis_files(self, root_directory: str) -> List[Document]:
        """
        Recursively finds and processes all markdown files for the openBIS source.
        """
        if not os.path.isdir(root_directory):
            logger.error(
                f"Error: The specified directory does not exist: {root_directory}"
            )
            return []

        all_chunks = []
        for dirpath, _, filenames in os.walk(root_directory):
            for filename in filenames:
                if filename.endswith(".md"):
                    file_path = os.path.join(dirpath, filename)
                    chunks = self._chunk_openbis_document(file_path, root_directory)
                    all_chunks.extend(chunks)

        return all_chunks

    @staticmethod
    def _create_and_persist_vectordb(chunks: List[Document], persist_directory: str):
        if not chunks:
            logger.warning("No chunks to process. Vector database will not be created.")
            return

        filtered_chunks = filter_complex_metadata(chunks)

        logger.info("Initializing embedding model...")
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        logger.info("-> Model: nomic-embed-text")

        logger.info(f"Creating/updating vector database at '{persist_directory}'...")
        logger.info(
            f"This may take a while, embedding {len(filtered_chunks)} chunks..."
        )

        Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},  # cosine distance.
        )
        logger.info("-> Vector database processing complete.")

    @staticmethod
    def export_chunks(chunks: List[Document], output_dir: str):
        """Exports the list of Document chunks to JSON, CSV, and JSONL files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_to_export = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        # Export to JSON
        json_path = os.path.join(output_dir, "chunks_openbis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                data_to_export, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder
            )
        logger.info(f"\nSuccessfully exported {len(chunks)} chunks to {json_path}")

        # Export to CSV
        csv_path = os.path.join(output_dir, "chunks_openbis.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            all_meta_keys = set().union(*(d["metadata"].keys() for d in data_to_export))

            preferred_order = [
                "id",
                "origin",
                "section",
                "source",
                "url",
                "title",
                "page_content",
            ]
            remaining_keys = sorted(list(all_meta_keys - set(preferred_order)))
            fieldnames = preferred_order + remaining_keys

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in data_to_export:
                row = {"page_content": item["page_content"]}
                row.update(item["metadata"])
                writer.writerow(row)
        logger.info(f"Successfully exported {len(chunks)} chunks to {csv_path}")

        # Export to JSONL
        jsonl_path = os.path.join(output_dir, "chunks_openbis.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in data_to_export:
                f.write(
                    json.dumps(item, ensure_ascii=False, cls=CustomJSONEncoder) + "\n"
                )
        logger.info(f"Successfully exported {len(chunks)} chunks to {jsonl_path}")


if __name__ == "__main__":
    # Default paths for standalone execution
    ROOT_DIRECTORY = "./data/raw/openbis"
    OUTPUT_DIRECTORY = "./data/processed/openbis"
    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"

    # Instantiate the processor and run the pipeline
    processor = OpenBisProcessor(
        root_directory=ROOT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    processor.process()
