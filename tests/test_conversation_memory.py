# tests/test_conversation_memory.py

import os
from pathlib import Path

import pytest

# Make sure to import the class from your source code
from desi.query.conversation_engine import ConversationMemory


# A pytest fixture to create and clean up the test database for each test
@pytest.fixture
def memory_db():
    """Provides a temporary ConversationMemory instance and cleans up its database file."""
    db_path = "test_conversation_memory.db"

    # Setup: Ensure no old DB exists before the test
    if os.path.exists(db_path):
        os.remove(db_path)

    memory = ConversationMemory(db_path=db_path)

    # 'yield' passes the memory object to the test function
    yield memory

    # Teardown: Clean up the database file after the test is done
    if os.path.exists(db_path):
        os.remove(db_path)


def test_add_and_get_messages(memory_db):
    """Test adding messages and retrieving them in chronological order."""
    session_id = "test_session_1"

    memory_db.add_message(
        session_id, "user", "Hello, what is openBIS?", 15, {"source": "test"}
    )
    memory_db.add_message(
        session_id, "assistant", "openBIS is a data management platform.", 25
    )

    messages = memory_db.get_recent_messages(session_id)

    # Use assertions to check correctness
    assert len(messages) == 2
    assert messages[0]["type"] == "user"
    assert messages[0]["content"] == "Hello, what is openBIS?"
    assert messages[0]["metadata"] == {"source": "test"}
    assert messages[1]["type"] == "assistant"


def test_token_counting(memory_db):
    """Test the total token counting logic."""
    session_id = "test_session_2"

    memory_db.add_message(session_id, "user", "Message 1", 10)
    memory_db.add_message(session_id, "assistant", "Response 1", 20)
    memory_db.add_message(session_id, "user", "Message 2", 15)

    total_tokens = memory_db.get_total_tokens(session_id)

    assert total_tokens == 10 + 20 + 15


def test_session_isolation(memory_db):
    """Test that different sessions do not interfere with each other."""
    session_id_1 = "session_A"
    session_id_2 = "session_B"

    memory_db.add_message(session_id_1, "user", "Message for A", 10)
    memory_db.add_message(session_id_2, "user", "Message for B", 12)

    messages_A = memory_db.get_recent_messages(session_id_1)
    messages_B = memory_db.get_recent_messages(session_id_2)

    assert len(messages_A) == 1
    assert messages_A[0]["content"] == "Message for A"
    assert len(messages_B) == 1
    assert messages_B[0]["content"] == "Message for B"
    assert memory_db.get_total_tokens(session_id_1) == 10


def test_clear_session(memory_db):
    """Test clearing a session's history."""
    session_id = "test_session_3"

    memory_db.add_message(session_id, "user", "This will be deleted.", 10)
    memory_db.clear_session(session_id)

    messages = memory_db.get_recent_messages(session_id)

    assert len(messages) == 0
    assert memory_db.get_total_tokens(session_id) == 0


def test_message_limit(memory_db):
    """Test that the message limit is respected."""
    session_id = "test_session_4"

    for i in range(10):
        memory_db.add_message(session_id, "user", f"Message {i}", 5)

    # Request only the 5 most recent messages
    messages = memory_db.get_recent_messages(session_id, limit=5)

    assert len(messages) == 5
    # The last message retrieved should be the last one added
    assert messages[-1]["content"] == "Message 9"
