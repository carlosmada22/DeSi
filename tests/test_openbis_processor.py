# Make sure the src directory is in the path for imports
import sys
from pathlib import Path
from unittest.mock import mock_open

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from desi.processor.openbis_processor import (
    ContentChunker,
    Document,
    OpenBisProcessor,
)

# --- Tests for the Cleaning Function ---


def test_clean_removes_permalink():
    """Tests that the [] permalink artifact is removed."""
    dirty = '## My Header[](#my-header "Permalink to this heading")'
    clean = "## My Header"
    assert OpenBisProcessor._clean_markdown_content(dirty) == clean


def test_clean_dedents_code_block():
    """Tests that indented code blocks are properly dedented."""
    dirty = "    // This is some code\n    if (true) {\n        // more code\n    }"
    clean = "// This is some code\nif (true) {\n    // more code\n}"
    assert OpenBisProcessor._clean_markdown_content(dirty) == clean


def test_clean_handles_nbsp_and_dedents():
    """Tests that non-breaking spaces are handled correctly, allowing dedent to work."""
    dirty = "    line 1\n\u00a0\n    line 2"  # \u00a0 is a non-breaking space
    clean = "line 1\n\nline 2"
    assert OpenBisProcessor._clean_markdown_content(dirty) == clean


def test_clean_collapses_newlines():
    """Tests that more than two newlines are collapsed."""
    dirty = "Paragraph 1\n\n\n\nParagraph 2"
    clean = "Paragraph 1\n\nParagraph 2"
    assert OpenBisProcessor._clean_markdown_content(dirty) == clean


# --- Tests for the Chunking Logic ---


def test_content_chunker_splits_correctly():
    """Tests the openBIS header-aware chunking logic."""
    chunker = ContentChunker(min_chunk_size=100, max_chunk_size=200)
    long_content = (
        "## Section Alpha\nThis is content for the first section. It is long enough to be its own chunk probably."
        "\n\n### Subsection A\nMore content here.\n\n"
        "## Section Beta\nThis is the second section. It should definitely start a new chunk because it is a new H2."
    )

    chunks = chunker.chunk_content(long_content)

    assert len(chunks) == 2
    assert "## Section Alpha" in chunks[0]
    assert "## Section Beta" in chunks[1]


def test_content_chunker_keeps_short_content_as_one_chunk():
    """Tests that short documents are not split."""
    chunker = ContentChunker()
    short_content = "## A Title\n\nJust a little bit of text."
    chunks = chunker.chunk_content(short_content)
    assert len(chunks) == 1
    assert chunks[0] == short_content


# --- Integration Test for the Main Function ---


def test_chunk_openbis_document_integration(tmp_path):
    """Tests the full processing of a single openBIS file, including metadata enrichment."""
    # Create a fake directory and file
    root_dir = tmp_path
    md_file = (
        root_dir
        / "en_20.10.0-11_user-documentation_general-users_managing-lab-stocks.md"
    )
    md_file.write_text("## Managing Lab Stocks\n\nThis is some sample content.")

    processor = OpenBisProcessor(
        root_directory=str(root_dir),
        output_directory="dummy_output",
        chroma_persist_directory="dummy_chroma",
    )

    chunks = processor._chunk_openbis_document(str(md_file))

    assert len(chunks) == 1
    chunk = chunks[0]

    # Verify metadata
    assert chunk.metadata["origin"] == "openbis"
    assert chunk.metadata["section"] == "User Documentation"
    assert chunk.metadata["source"].endswith("managing-lab-stocks.md")
    assert chunk.metadata["id"].startswith(
        "openbis-en_20.10.0-11_user-documentation_general-users_managing-lab-stocks-"
    )
    assert chunk.metadata["title"] == "Managing lab stocks"
    assert "https://openbis.readthedocs.io/" in chunk.metadata["url"]

    # Verify content
    assert "## Managing Lab Stocks" in chunk.page_content
