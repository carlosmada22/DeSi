# tests/test_conversation_memory.py

import os
import sys
from pathlib import Path

import pytest

# Add the src directory to the path to ensure imports work from the root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# --- CHANGE 1: Import the correct class name ---
from desi.query.conversation_engine import SqliteConversationMemory


# A pytest fixture to create and clean up the test database for each test
@pytest.fixture
def memory_db():
    """Provides a temporary SqliteConversationMemory instance and cleans up its database file."""
    db_path = "test_conversation_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # --- CHANGE 2: Instantiate the correct class ---
    memory = SqliteConversationMemory(db_path=db_path, history_limit=20)

    yield memory

    if os.path.exists(db_path):
        os.remove(db_path)


def test_add_and_get_messages(memory_db):
    """Test adding messages and retrieving them in chronological order."""
    session_id = "test_session_1"

    # --- CHANGE 3: Use the correct method signature: (session_id, role, content) ---
    memory_db.add_message(session_id, "user", "Hello, what is openBIS?")
    memory_db.add_message(
        session_id, "assistant", "openBIS is a data management platform."
    )

    messages = memory_db.get_recent_messages(session_id)

    assert len(messages) == 2
    # --- CHANGE 4: The key in the returned dictionary is 'role', not 'type' ---
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, what is openBIS?"
    assert messages[1]["role"] == "assistant"


def test_session_isolation(memory_db):
    """Test that different sessions do not interfere with each other."""
    session_id_1 = "session_A"
    session_id_2 = "session_B"

    memory_db.add_message(session_id_1, "user", "Message for A")
    memory_db.add_message(session_id_2, "user", "Message for B")

    messages_A = memory_db.get_recent_messages(session_id_1)
    messages_B = memory_db.get_recent_messages(session_id_2)

    assert len(messages_A) == 1
    assert messages_A[0]["content"] == "Message for A"
    assert len(messages_B) == 1
    assert messages_B[0]["content"] == "Message for B"


def test_clear_session(memory_db):
    """Test clearing a session's history."""
    session_id = "test_session_3"

    memory_db.add_message(session_id, "user", "This will be deleted.")
    memory_db.clear_session(session_id)

    messages = memory_db.get_recent_messages(session_id)

    assert len(messages) == 0


def test_message_limit_is_respected():
    """Test that the history_limit from the constructor is respected."""
    db_path = "test_limit_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # --- CHANGE 5: The limit is set during initialization, not when calling the method ---
    # Instantiate with a small limit for this specific test
    memory_with_limit = SqliteConversationMemory(db_path=db_path, history_limit=5)

    session_id = "test_session_4"

    for i in range(10):
        memory_with_limit.add_message(session_id, "user", f"Message {i}")

    messages = memory_with_limit.get_recent_messages(session_id)

    # The class should only return the last 5 messages
    assert len(messages) == 5
    # The first message in the returned list should be the 5th one added (index 5)
    assert messages[0]["content"] == "Message 5"
    # The last message in the returned list should be the last one added (index 9)
    assert messages[-1]["content"] == "Message 9"

    if os.path.exists(db_path):
        os.remove(db_path)
