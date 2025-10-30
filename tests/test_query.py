# Ensure the source directory is in the Python path for imports
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Add the src directory to the path to ensure imports work from the root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# We need the Document class to create mock data for our tests
from langchain_core.documents import Document

from desi.query.query import RAGQueryEngine

# --- Mocks and Fixtures ---

# A fake prompt template to be used by the mock file system
FAKE_PROMPT_TEMPLATE = "History: {history_str}\nContext: {context_str}\nQuery: {query}"


@pytest.fixture
def mock_docs_and_scores():
    """Provides a list of mock (Document, score) tuples for mocking ChromaDB results."""
    return [
        (
            Document(
                page_content="Content from openBIS doc.", metadata={"origin": "openbis"}
            ),
            0.8,
        ),
        (
            Document(
                page_content="Content from dswiki doc 1.", metadata={"origin": "dswiki"}
            ),
            0.75,
        ),
        (
            Document(
                page_content="Low relevance content.", metadata={"origin": "openbis"}
            ),
            0.2,
        ),  # This should be filtered out
        (
            Document(
                page_content="Content from dswiki doc 2.", metadata={"origin": "dswiki"}
            ),
            0.6,
        ),
        (
            Document(
                page_content="More openBIS content.", metadata={"origin": "openbis"}
            ),
            0.5,
        ),
    ]


@pytest.fixture
def rag_engine(mocker):
    """
    Provides a fully initialized RAGQueryEngine instance with all external
    dependencies (Ollama, Chroma, file I/O) mocked out.
    """
    # Mock the Ollama and ChromaDB components
    mocker.patch("desi.query.query.OllamaEmbeddings", return_value=MagicMock())
    mock_chroma_instance = MagicMock()
    mocker.patch("desi.query.query.Chroma", return_value=mock_chroma_instance)

    mock_llm_instance = MagicMock()
    # Mock the .invoke() method on the LLM to return a mock response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is the generated answer from the LLM."
    mock_llm_instance.invoke.return_value = mock_llm_response
    mocker.patch("desi.query.query.ChatOllama", return_value=mock_llm_instance)

    # Mock the file system to avoid needing a real prompt file
    mocker.patch("builtins.open", mock_open(read_data=FAKE_PROMPT_TEMPLATE))

    # Mock the global availability flag to ensure the engine initializes
    mocker.patch("desi.query.query.OLLAMA_AVAILABLE", True)

    # Instantiate the class under test
    engine = RAGQueryEngine(
        chroma_persist_directory="/fake/dir",
        prompt_template_path="/fake/prompt.md",
        relevance_score_threshold=0.3,
        dswiki_boost=0.15,
    )
    # Attach the mocks to the instance for easy access in tests
    engine.vector_store = mock_chroma_instance
    engine.llm = mock_llm_instance
    return engine


# --- Unit Tests ---


def test_initialization_no_ollama(mocker):
    """Tests that the engine handles initialization gracefully when Ollama is not available."""
    mocker.patch("desi.query.query.OLLAMA_AVAILABLE", False)
    mocker.patch("builtins.open", mock_open(read_data=FAKE_PROMPT_TEMPLATE))

    engine = RAGQueryEngine(
        chroma_persist_directory="/fake/dir", prompt_template_path="/fake/prompt.md"
    )

    assert engine.vector_store is None
    assert engine.llm is None


def test_retrieve_relevant_chunks_logic(rag_engine, mock_docs_and_scores):
    """
    Tests the core logic of retrieving, filtering, boosting, and re-ranking chunks.
    """
    # Configure the mock Chroma vector store to return our predefined docs and scores
    rag_engine.vector_store.similarity_search_with_relevance_scores.return_value = (
        mock_docs_and_scores
    )

    query = "test query"
    top_k = 3
    final_chunks = rag_engine.retrieve_relevant_chunks(query, top_k=top_k)

    # 1. Check that the final list has the correct size (top_k)
    assert len(final_chunks) == top_k

    # 2. Check that the low-scoring doc was filtered out and is not in the final list
    assert "Low relevance content." not in [doc.page_content for doc in final_chunks]

    # 3. Check that the re-ranking worked as expected.
    # The first dswiki doc (score 0.75) should be boosted to 0.90 (0.75 + 0.15).
    # This should now be the highest-scoring document, even higher than the original 0.8 openBIS doc.
    assert final_chunks[0].page_content == "Content from dswiki doc 1."
    assert final_chunks[1].page_content == "Content from openBIS doc."

    # 4. The second dswiki doc (0.6) is boosted to 0.75, making it third.
    assert final_chunks[2].page_content == "Content from dswiki doc 2."


def test_create_prompt_with_context_and_history(rag_engine, mock_docs_and_scores):
    """Tests that the prompt is formatted correctly with all components."""
    # We only need the Document part of the tuple
    relevant_chunks = [doc for doc, score in mock_docs_and_scores[:2]]
    query = "What is openBIS?"
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    prompt = rag_engine._create_prompt(query, relevant_chunks, history)

    # Check that all parts are present
    assert "History: User: Hello\nAssistant: Hi there!" in prompt
    assert "Context: --- Context Chunk 1" in prompt
    assert "Content from openBIS doc." in prompt
    assert "Content from dswiki doc 1." in prompt
    assert "Query: What is openBIS?" in prompt


def test_generate_answer_cleans_think_tags(rag_engine):
    """Tests that the LLM response is correctly cleaned of <think> tags."""
    # Configure the mock LLM to return a response with think tags
    rag_engine.llm.invoke.return_value.content = "<think>I should formulate the answer carefully.</think>The final answer is here."

    prompt = "test prompt"
    answer = rag_engine.generate_answer(prompt)

    assert answer == "The final answer is here."
    assert "<think>" not in answer


def test_query_orchestration(mocker, rag_engine):
    """
    Tests that the main `query` method correctly calls its helper methods in sequence.
    """
    # Spy on the internal methods of the already-mocked rag_engine instance
    mocker.patch.object(rag_engine, "retrieve_relevant_chunks", return_value=[])
    mocker.patch.object(rag_engine, "_create_prompt", return_value="final prompt")
    mocker.patch.object(rag_engine, "generate_answer", return_value="final answer")

    query = "test query"
    answer, chunks = rag_engine.query(query)

    # Assert that the main components of the pipeline were called
    rag_engine.retrieve_relevant_chunks.assert_called_once_with(query, top_k=5)
    rag_engine._create_prompt.assert_called_once()
    rag_engine.generate_answer.assert_called_once_with("final prompt")

    # Assert that the final output is correct
    assert answer == "final answer"
