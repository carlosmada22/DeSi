# Ensure the source directory is in the Python path for imports
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

# Add the src directory to the path to ensure imports work from the root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document

from desi.query.conversation_engine import (
    ChatbotEngine,
    GraphState,
    SqliteConversationMemory,
)
from desi.query.query import RAGQueryEngine

# --- Tests for SqliteConversationMemory ---


@pytest.fixture
def mock_sqlite_cursor(mocker):
    """
    Mocks sqlite3.connect to always return a connection that, in turn,
    always returns the *same single mock cursor*. This is the key to the fix.
    """
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mocker.patch("sqlite3.connect", return_value=mock_conn)
    return mock_cursor


def test_memory_add_message(mock_sqlite_cursor):
    """Tests that a message is correctly inserted into the mock database."""
    memory = SqliteConversationMemory(db_path=":memory:")
    assert mock_sqlite_cursor.execute.call_count == 2

    memory.add_message("session123", "user", "Hello")

    assert mock_sqlite_cursor.execute.call_count == 3
    mock_sqlite_cursor.execute.assert_called_with(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        ("session123", "user", "Hello"),
    )


def test_memory_get_recent_messages(mock_sqlite_cursor):
    """Tests that messages are correctly retrieved and formatted."""
    # The code's `reversed()` call will then flip it back to chronological for the final output.
    mock_sqlite_cursor.fetchall.return_value = [
        ("assistant", "Hi there"),  # Most recent message first
        ("user", "Hello"),  # Oldest message last
    ]

    memory = SqliteConversationMemory(db_path=":memory:")
    history = memory.get_recent_messages("session123")

    # The assertion now correctly expects chronological order.
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    mock_sqlite_cursor.fetchall.assert_called_once()


# --- Tests for ChatbotEngine ---


@pytest.fixture
def chatbot_engine(mocker):
    """Provides a fully mocked ChatbotEngine instance for testing its nodes."""
    mock_rag_engine = MagicMock(spec=RAGQueryEngine)
    mock_memory = MagicMock(spec=SqliteConversationMemory)
    mock_rewrite_llm = MagicMock(spec=ChatOllama)

    mocker.patch.object(ChatbotEngine, "_build_graph", return_value=MagicMock())

    engine = ChatbotEngine(
        rag_engine=mock_rag_engine,
        memory=mock_memory,
        rewrite_llm=mock_rewrite_llm,
    )
    engine.mock_rag_engine = mock_rag_engine
    engine.mock_memory = mock_memory
    engine.mock_rewrite_llm = mock_rewrite_llm

    return engine


def test_rewrite_query_node_with_history(chatbot_engine):
    """Tests that the query is rewritten when conversation history exists."""
    state = GraphState(user_query="What about them?", session_id="s1")

    chatbot_engine.mock_memory.get_recent_messages.return_value = [
        {"role": "user", "content": "Tell me about documents."}
    ]
    chatbot_engine.mock_rewrite_llm.invoke.return_value.content = (
        "What about documents?"
    )

    result = chatbot_engine.rewrite_query_node(state)

    assert result["rewritten_query"] == "What about documents?"
    chatbot_engine.mock_rewrite_llm.invoke.assert_called_once()


def test_rewrite_query_node_no_history(chatbot_engine):
    """Tests that the query is NOT rewritten when there is no history."""
    state = GraphState(user_query="What are documents?", session_id="s1")

    chatbot_engine.mock_memory.get_recent_messages.return_value = []

    result = chatbot_engine.rewrite_query_node(state)

    assert result["rewritten_query"] == "What are documents?"
    chatbot_engine.mock_rewrite_llm.invoke.assert_not_called()


def test_call_rag_engine_node(chatbot_engine):
    """Tests that the RAG engine is called with the rewritten query and history."""
    state = GraphState(
        rewritten_query="Standalone question", session_id="s1", user_query="follow-up"
    )

    mock_history = [{"role": "user", "content": "Previous message"}]
    chatbot_engine.mock_memory.get_recent_messages.return_value = mock_history

    mock_sources = [Document(page_content="text", metadata={"source": "doc1.md"})]
    chatbot_engine.mock_rag_engine.query.return_value = ("RAG Answer", mock_sources)

    result = chatbot_engine.call_rag_engine(state)

    assert result["response"] == "RAG Answer"
    assert result["sources"] == [{"source": "doc1.md"}]

    chatbot_engine.mock_rag_engine.query.assert_called_once_with(
        query="Standalone question", conversation_history=mock_history
    )


def test_update_memory_node(chatbot_engine):
    """Tests that user query and assistant response are saved to memory."""
    state = GraphState(
        session_id="s1",
        user_query="My Question",
        response="Assistant Answer",
    )

    chatbot_engine.update_memory(state)

    expected_calls = [
        call("s1", "user", "My Question"),
        call("s1", "assistant", "Assistant Answer"),
    ]
    chatbot_engine.mock_memory.add_message.assert_has_calls(expected_calls)


def test_chat_method_orchestration(mocker, chatbot_engine):
    """
    Tests that the main `chat` method correctly invokes the compiled graph.
    """
    final_state = {
        "response": "Final graph response",
        "sources": [{"source": "final_doc.md"}],
    }
    mock_graph_invoke = mocker.patch.object(
        chatbot_engine.graph, "invoke", return_value=final_state
    )

    user_input = "Test input"
    session_id = "s1"

    response, sources = chatbot_engine.chat(user_input, session_id)

    mock_graph_invoke.assert_called_once()
    initial_state_arg = mock_graph_invoke.call_args[0][0]
    assert initial_state_arg["user_query"] == user_input
    assert initial_state_arg["session_id"] == session_id

    assert response == "Final graph response"
    assert sources == [{"source": "final_doc.md"}]
