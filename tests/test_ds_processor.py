# Make sure the src directory is in the path for imports
# This is often handled by project setup (e.g., pyproject.toml) but can be added manually if needed
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from desi.processor.ds_processor import (
    Document,
    _parse_mermaid_logic,
    chunk_document,
    clean_chunk_content,
    load_and_preprocess_file,
    split_markdown_by_structure,
)


# --- Test Data Fixtures ---
@pytest.fixture
def sample_md_with_yaml():
    return """---
title: Test Document
author: Pytest
date: 2023-10-27
---
## Section 1
This is the first paragraph. It is simple.

This is the second paragraph.
"""


@pytest.fixture
def sample_md_no_yaml():
    return "## Just Content\n\nThis is a document without any YAML frontmatter."


@pytest.fixture
def sample_mermaid_diagram():
    return """
Some introductory text.
```mermaid
graph TD;
    A([Start Here]);
    B([Process Step]);
    C([End Here]);
    A-->B;
    B-->C;
```
More text after.
"""


# --- Tests for Individual Functions ---


def test_load_and_preprocess_file_with_yaml(mocker, sample_md_with_yaml):
    """Tests that YAML frontmatter is correctly parsed."""
    mocker.patch("builtins.open", mock_open(read_data=sample_md_with_yaml))

    content, metadata = load_and_preprocess_file("fake/path.md")

    assert (
        content
        == "## Section 1\nThis is the first paragraph. It is simple.\n\nThis is the second paragraph."
    )
    assert metadata["title"] == "Test Document"
    assert metadata["author"] == "Pytest"
    assert "full_path" in metadata


def test_load_and_preprocess_file_no_yaml(mocker, sample_md_no_yaml):
    """Tests that files without YAML are handled gracefully."""
    mocker.patch("builtins.open", mock_open(read_data=sample_md_no_yaml))

    content, metadata = load_and_preprocess_file("fake/path.md")

    assert (
        content == "## Just Content\n\nThis is a document without any YAML frontmatter."
    )
    assert metadata == {"full_path": "fake/path.md"}


def test_parse_mermaid_logic():
    """Tests the conversion of a mermaid diagram to text."""
    # FIX: The node text is now quoted, matching the parser's expectation.
    diagram = '```mermaid graph TB;\nA(["Start Process"]);\nB(["End Process"]);\nA-->B;'
    expected = "The process is as follows:\nStep 1: Start Process\nStep 2: End Process"

    result = _parse_mermaid_logic(diagram)
    assert result.strip() == expected


def test_clean_chunk_content_removes_html():
    """Tests that HTML tags are stripped from content."""
    dirty_content = "This is <b>bold</b> and <i>italic</i>. <!-- comment -->"
    expected_content = "This is bold and italic."

    result = clean_chunk_content(dirty_content)
    assert result == expected_content


def test_split_markdown_by_structure_creates_chunks():
    """Tests the structural chunking logic."""
    long_content = (
        "## Section 1\n\nThis is the first paragraph.\n\n"
        "### Subsection 1.1\n\nThis is more content that belongs to the first section.\n\n"
        "## Section 2\n\nThis is the start of the second section, which should be a new chunk."
    )
    metadata = {"origin": "dswiki"}

    chunks = split_markdown_by_structure(long_content, metadata, min_chunk_size=50)

    assert len(chunks) == 2
    assert "## Section 1" in chunks[0].page_content
    assert "## Section 2" in chunks[1].page_content
    assert chunks[0].metadata["origin"] == "dswiki"


def test_chunk_document_integration(tmp_path):
    """An integration test for the main document chunking function."""
    # Create a fake directory structure
    root_dir = tmp_path
    section_dir = root_dir / "use_cases"
    section_dir.mkdir()
    md_file = section_dir / "my-test-file.md"
    md_file.write_text(
        "---\ntitle: Integration Test\n---\n## A Section\n\nSome content here that is definitely long enough to pass the minimum length check after being processed."
    )

    chunks = chunk_document(str(md_file), str(root_dir))

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.metadata["origin"] == "dswiki"
    assert chunk.metadata["section"] == "Use Cases"
    assert chunk.metadata["source"] == "use_cases/my-test-file.md"

    assert chunk.metadata["id"] == "dswiki-use_cases-my-test-file"

    assert (
        "Integration Test" in chunk.metadata["title"]
    )  # Assuming title is added in metadata
    assert "Some content here" in chunk.page_content
