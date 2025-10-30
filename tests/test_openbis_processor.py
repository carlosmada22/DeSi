# Make sure the src directory is in the path for imports
import os
import sys
from pathlib import Path
from unittest.mock import call

import pytest

# Add the src directory to the path to ensure imports work from the root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document

from desi.processor.openbis_processor import ContentChunker, OpenBisProcessor

# --- Tests for the Cleaning Function ---


@pytest.mark.parametrize(
    "dirty_input, expected_clean",
    [
        ('## My Header[](#my-header "Permalink to this heading")', "## My Header"),
        ("    // code block", "// code block"),
        ("    line 1\n\u00a0\n    line 2", "line 1\n\nline 2"),
        ("Paragraph 1\n\n\n\nParagraph 2", "Paragraph 1\n\nParagraph 2"),
        ("This is a clean line.", "This is a clean line."),
        ("Title[](...)", "Title[](...)"),
    ],
)
def test_clean_markdown_content(dirty_input, expected_clean):
    """Tests various scenarios for the markdown cleaning logic."""
    assert OpenBisProcessor._clean_markdown_content(dirty_input) == expected_clean


# --- Tests for the ContentChunker Logic ---


@pytest.fixture
def chunker():
    """Provides a default ContentChunker instance."""
    return ContentChunker(min_chunk_size=100, max_chunk_size=300)


def test_content_chunker_splits_by_h2(chunker):
    """Tests that content is split correctly at H2 (##) headers."""
    long_content = (
        "## Section Alpha\nThis is content for the first section. It is long enough to be its own chunk and should not be merged with the next one."
        "\n\n### Subsection A\nMore content here that belongs to Section Alpha.\n\n"
        "## Section Beta\nThis is the second section. It should definitely start a new chunk because it is a new H2 header."
    )
    chunks = chunker.chunk_content(long_content)
    assert len(chunks) == 2
    assert chunks[0].startswith("## Section Alpha")
    assert "Subsection A" in chunks[0]
    assert chunks[1].startswith("## Section Beta")


def test_content_chunker_keeps_short_content_as_one_chunk(chunker):
    """Tests that short documents are not split, even if they have headers."""
    short_content = "## A Title\n\nJust a little bit of text here."
    chunks = chunker.chunk_content(short_content)
    assert len(chunks) == 1
    assert chunks[0] == short_content


def test_content_chunker_splits_long_section_while_preserving_context(chunker):
    """Tests that a section exceeding max_chunk_size is split, and the header context is re-added."""
    long_section = (
        "## Very Long Section\n\n"
        "This is the first paragraph of a very long section. It contains a lot of text to ensure it will exceed the maximum chunk size of 300 characters defined in the fixture. We will keep adding sentences to pad it out. More text here to fill space. And even more text. \n\n"
        "This is the second paragraph which should definitely be in a new chunk. By placing this text here, we force the chunker to make a decision and split the content, and the test will verify that the '## Very Long Section' header is prepended to this new chunk."
    )
    chunks = chunker.chunk_content(long_section)
    assert len(chunks) == 2
    assert chunks[0].startswith("## Very Long Section")
    assert "first paragraph" in chunks[0]
    assert "second paragraph" not in chunks[0]
    assert chunks[1].startswith("## Very Long Section")
    assert "second paragraph" in chunks[1]


# --- Tests for the Main Processor Class ---


@pytest.fixture
def processor(tmp_path):
    """Provides an OpenBisProcessor instance initialized with a temporary directory."""
    return OpenBisProcessor(
        root_directory=str(tmp_path),
        output_directory=str(tmp_path / "processed"),
        chroma_persist_directory=str(tmp_path / "vectordb"),
    )


def test_chunk_openbis_document_integration(processor, tmp_path):
    """
    Tests the full processing of a single file, including cleaning, metadata enrichment, and chunking.
    """
    file_name = "en_20.10.0-11_user-documentation_general-users_managing-lab-stocks.md"
    md_file = tmp_path / file_name

    md_file.write_text(
        '## Managing Lab Stocks[](#my-header "Permalink to this heading")\n\nThis is some sample content.',
        encoding="utf-8",
    )

    chunks = processor._chunk_openbis_document(str(md_file), str(tmp_path))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["origin"] == "openbis"
    assert chunk.metadata["section"] == "User Documentation"
    assert chunk.metadata["source"] == file_name.replace("\\", "/")
    assert (
        chunk.metadata["id"]
        == "openbis-en_20.10.0-11_user-documentation_general-users_managing-lab-stocks-0"
    )
    assert "## Managing Lab Stocks" in chunk.page_content
    assert "[]" not in chunk.page_content


def test_process_all_openbis_files(mocker, processor):
    """Tests the file discovery and processing orchestration."""
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("os.walk").return_value = [
        ("/fake_root", (), ("doc1.md", "doc2.txt", "doc3.md")),
    ]
    mock_chunker = mocker.patch.object(
        processor, "_chunk_openbis_document", return_value=[Document(page_content="")]
    )

    result_chunks = processor._process_all_openbis_files("/fake_root")

    assert mock_chunker.call_count == 2
    assert len(result_chunks) == 2
    expected_calls = [
        call(os.path.join("/fake_root", "doc1.md"), "/fake_root"),
        call(os.path.join("/fake_root", "doc3.md"), "/fake_root"),
    ]
    mock_chunker.assert_has_calls(expected_calls, any_order=True)


def test_process_method_orchestration(mocker, processor):
    """
    Tests that the main `process` method calls its helpers in the correct sequence.
    """
    mock_process_files = mocker.patch.object(
        processor,
        "_process_all_openbis_files",
        return_value=[Document(page_content="chunk", metadata={})],
    )
    mock_export = mocker.patch.object(processor, "_export_chunks")
    mock_vectordb = mocker.patch.object(processor, "_create_and_persist_vectordb")

    processor.process()

    mock_process_files.assert_called_once()
    mock_export.assert_called_once()
    mock_vectordb.assert_called_once()
