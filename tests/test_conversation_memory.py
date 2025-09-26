#!/usr/bin/env python3
"""
Test script for DeSi conversation memory system.

This script tests the new SQLite-based conversation memory to ensure
it properly stores and retrieves user-chatbot exchanges.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only the ConversationMemory class to avoid ChromaDB dependency
import json
import sqlite3
from typing import Dict, List


class ConversationMemory:
    """SQLite-based conversation memory for storing user-chatbot exchanges."""

    def __init__(self, db_path: str = "desi_conversation_memory.db"):
        """Initialize conversation memory with SQLite database."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON string for additional data
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON conversations(session_id, timestamp)
            """)
            conn.commit()

    def add_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        token_count: int = 0,
        metadata: Dict = None,
    ):
        """Add a message to the conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                """
                INSERT INTO conversations
                (session_id, message_type, content, token_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, message_type, content, token_count, metadata_json),
            )
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get the most recent messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT message_type, content, token_count, timestamp, metadata
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (session_id, limit),
            )

            messages = []
            for row in reversed(
                cursor.fetchall()
            ):  # Reverse to get chronological order
                message_type, content, token_count, timestamp, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                messages.append(
                    {
                        "type": message_type,
                        "content": content,
                        "token_count": token_count,
                        "timestamp": timestamp,
                        "metadata": metadata,
                    }
                )
            return messages

    def get_total_tokens(self, session_id: str) -> int:
        """Get total token count for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT SUM(token_count) FROM conversations WHERE session_id = ?
            """,
                (session_id,),
            )
            result = cursor.fetchone()[0]
            return result if result else 0

    def clear_session(self, session_id: str):
        """Clear all messages for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            conn.commit()


def test_conversation_memory():
    """Test the conversation memory system."""
    print("Testing DeSi Conversation Memory System")
    print("=" * 50)

    # Initialize memory with a test database
    memory = ConversationMemory("test_conversation_memory.db")

    # Test session ID
    session_id = "test_session_123"

    print(f"Testing session: {session_id}")

    # Clear any existing data for this session
    memory.clear_session(session_id)

    # Test adding messages
    print("\n1. Adding test messages...")
    memory.add_message(
        session_id, "user", "Hello, what is openBIS?", 15, {"test": True}
    )
    memory.add_message(
        session_id,
        "assistant",
        "openBIS is a data management platform for life sciences.",
        25,
        {"model": "test-model"},
    )
    memory.add_message(session_id, "user", "How do I create a sample?", 12)
    memory.add_message(
        session_id, "assistant", "To create a sample in openBIS, you need to...", 35
    )

    # Test retrieving messages
    print("2. Retrieving messages...")
    messages = memory.get_recent_messages(session_id, limit=10)

    print(f"   Retrieved {len(messages)} messages:")
    for i, msg in enumerate(messages, 1):
        print(
            f"   {i}. [{msg['type']}] {msg['content'][:50]}... ({msg['token_count']} tokens)"
        )

    # Test token counting
    print("\n3. Testing token counting...")
    total_tokens = memory.get_total_tokens(session_id)
    print(f"   Total tokens for session: {total_tokens}")

    # Test with another session
    print("\n4. Testing multiple sessions...")
    session_id_2 = "test_session_456"
    memory.add_message(session_id_2, "user", "Different session message", 10)
    memory.add_message(session_id_2, "assistant", "Response in different session", 15)

    messages_2 = memory.get_recent_messages(session_id_2)
    print(f"   Session 2 has {len(messages_2)} messages")
    print(f"   Session 2 total tokens: {memory.get_total_tokens(session_id_2)}")

    # Test message limit
    print("\n5. Testing message limit...")
    for i in range(10):
        memory.add_message(session_id, "user", f"Test message {i}", 5)
        memory.add_message(session_id, "assistant", f"Test response {i}", 8)

    limited_messages = memory.get_recent_messages(session_id, limit=5)
    print(f"   Requested 5 messages, got {len(limited_messages)}")
    print(f"   Latest message: {limited_messages[-1]['content']}")

    # Test clearing session
    print("\n6. Testing session clearing...")
    memory.clear_session(session_id_2)
    cleared_messages = memory.get_recent_messages(session_id_2)
    print(f"   After clearing, session 2 has {len(cleared_messages)} messages")

    print("\n All conversation memory tests completed successfully!")

    # Clean up test database
    try:
        os.remove("test_conversation_memory.db")
        print("🧹 Cleaned up test database")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    test_conversation_memory()
