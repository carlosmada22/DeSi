#!/usr/bin/env python3
"""
Conversation engine for DeSi.

This module manages the conversational workflow on top of LangGraph, combining
retrieval-augmented generation, conversation memory, and the unified processor
vector store.
"""

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .query import DesiRAGQueryEngine

logger = logging.getLogger(__name__)


class ConversationMemory:
    """SQLite-based conversation memory for storing user-chatbot exchanges."""

    def __init__(self, db_path: str = "desi_conversation_memory.db"):
        """Initialize conversation memory with SQLite database."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON conversations(session_id, timestamp)
                """
            )
            conn.commit()

    def add_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        token_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
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

            messages: List[Dict] = []
            for row in reversed(cursor.fetchall()):
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
                "SELECT SUM(token_count) FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            result = cursor.fetchone()[0]
            return result if result else 0

    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            conn.commit()


class ConversationState(TypedDict):
    """State for the conversation graph."""

    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    rag_context: List[Dict]
    response: str
    session_id: str
    token_count: int


DEFAULT_HISTORY_LIMIT = 20


class DesiConversationEngine:
    """Conversation engine for DeSi with memory and RAG integration."""

    def __init__(
        self,
        db_path: str = "desi_vectordb",
        collection_name: str = "desi_docs",
        model: str = "gpt-oss:20b",
        memory_db_path: str = "desi_conversation_memory.db",
        retrieval_top_k: int = 5,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ):
        """
        Initialize the conversation engine.

        Args:
            db_path: Path to the ChromaDB database directory.
            collection_name: Name of the ChromaDB collection.
            model: Ollama model to use (when available).
            memory_db_path: Path to SQLite database for conversation memory.
            retrieval_top_k: Number of documentation chunks to retrieve per turn.
            history_limit: Number of historical turns to feed back into prompts.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.memory_db_path = memory_db_path
        self.retrieval_top_k = retrieval_top_k
        self.history_limit = history_limit

        self.rag_engine = DesiRAGQueryEngine(
            db_path=db_path,
            collection_name=collection_name,
            model=model,
        )
        logger.info(
            "Conversation engine initialised (Ollama available: %s)",
            self.rag_engine.ollama_available,
        )

        self.conversation_memory = ConversationMemory(memory_db_path)
        conn = sqlite3.connect(memory_db_path, check_same_thread=False)
        self.memory = SqliteSaver(conn=conn)
        self.graph = self._build_graph()

    def _conversation_history_for_prompt(self, session_id: str) -> List[Dict[str, str]]:
        """Return recent conversation history formatted for prompting."""
        raw_history = self.conversation_memory.get_recent_messages(
            session_id, limit=self.history_limit
        )
        history: List[Dict[str, str]] = []
        for item in raw_history:
            history.append(
                {
                    "role": item.get("type", "user"),
                    "content": item.get("content", ""),
                }
            )
        return history

    def _build_state_messages(
        self,
        conversation_history: List[Dict[str, str]],
        rag_chunks: List[Dict],
        user_query: str,
        assistant_answer: str,
    ) -> List[BaseMessage]:
        """Assemble LangChain message objects for graph state tracking."""
        messages: List[BaseMessage] = []

        system_message = SystemMessage(
            content=(
                "You are DeSi, a knowledgeable assistant specializing in openBIS "
                "and data store operations. Be friendly, clear, and ground every "
                "answer in the retrieved documentation context."
            )
        )
        messages.append(system_message)

        for item in conversation_history:
            role = (item.get("role") or "user").lower()
            content = item.get("content", "")
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        if rag_chunks:
            context_lines = []
            for idx, chunk in enumerate(rag_chunks, start=1):
                source_label = chunk.get("source_label") or chunk.get(
                    "source", "unknown"
                )
                title = chunk.get("title", "Untitled")
                score = chunk.get("similarity_score", 0.0)
                context_lines.append(
                    f"[Context {idx}] {source_label} · {title} (score {score:.3f})"
                )
            messages.append(
                SystemMessage(
                    content="Retrieved documentation context:\n"
                    + "\n".join(context_lines)
                )
            )

        messages.append(HumanMessage(content=user_query))
        messages.append(AIMessage(content=assistant_answer))
        return messages

    @staticmethod
    def _estimate_token_count(messages: List[BaseMessage]) -> int:
        """Rough token estimation based on whitespace-delimited words."""
        text = " ".join(
            msg.content for msg in messages if hasattr(msg, "content") and msg.content
        )
        return int(len(text.split()) * 1.3)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation flow."""

        def rag_agent(state: ConversationState) -> ConversationState:
            """RAG agent for documentation queries."""
            try:
                session_id = state.get("session_id") or "default"
                conversation_history = self._conversation_history_for_prompt(session_id)

                answer, relevant_chunks = self.rag_engine.query(
                    state["user_query"],
                    top_k=self.retrieval_top_k,
                    conversation_history=conversation_history,
                )

                cleaned_answer = self.clean_response(answer)
                state["response"] = cleaned_answer
                state["rag_context"] = relevant_chunks
                state["messages"] = self._build_state_messages(
                    conversation_history,
                    relevant_chunks,
                    state["user_query"],
                    cleaned_answer,
                )
                state["token_count"] = self._estimate_token_count(state["messages"])

                logger.info(
                    "Generated response for session %s (chunks=%d, tokens≈%d)",
                    session_id,
                    len(relevant_chunks),
                    state["token_count"],
                )
                return state
            except Exception as exc:
                logger.error(f"Error in RAG agent: {exc}")
                state["response"] = f"I encountered an error: {exc}"
                state["rag_context"] = []
                state["messages"] = []
                state["token_count"] = 0
                return state

        def update_conversation(state: ConversationState) -> ConversationState:
            """Persist the latest exchange in the conversation memory."""
            try:
                session_id = state.get("session_id") or "default"

                user_tokens = int(len(state["user_query"].split()) * 1.3)
                assistant_tokens = int(len(state["response"].split()) * 1.3)

                self.conversation_memory.add_message(
                    session_id=session_id,
                    message_type="user",
                    content=state["user_query"],
                    token_count=user_tokens,
                    metadata={
                        "rag_context_count": len(state.get("rag_context", [])),
                        "retrieval_top_k": self.retrieval_top_k,
                    },
                )

                self.conversation_memory.add_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=state["response"],
                    token_count=assistant_tokens,
                    metadata={
                        "model": self.model,
                        "rag_sources": [
                            chunk.get("source")
                            for chunk in state.get("rag_context", [])
                        ],
                    },
                )

                total_tokens = self.conversation_memory.get_total_tokens(session_id)
                state["token_count"] = total_tokens
                logger.info(
                    "Saved conversation turn. Session %s now has %d tokens.",
                    session_id,
                    total_tokens,
                )
                return state
            except Exception as exc:
                logger.error(f"Error updating conversation: {exc}")
                return state

        workflow = StateGraph(ConversationState)
        workflow.add_node("rag_agent", rag_agent)
        workflow.add_node("update_conversation", update_conversation)

        workflow.set_entry_point("rag_agent")
        workflow.add_edge("rag_agent", "update_conversation")
        workflow.add_edge("update_conversation", END)

        return workflow.compile(checkpointer=self.memory)

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id

    @staticmethod
    def clean_response(response: str) -> str:
        """Remove <think></think> tags from the response."""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
        return cleaned.strip()

    def chat(
        self, user_input: str, session_id: Optional[str] = None
    ) -> Tuple[str, str, Dict]:
        """
        Process a user input and return the response.

        Args:
            user_input: The user's message.
            session_id: Optional session ID for conversation continuity.

        Returns:
            Tuple of (response, session_id, metadata).
        """
        if not session_id:
            session_id = self.create_session()

        config = RunnableConfig(configurable={"thread_id": session_id})

        initial_state = ConversationState(
            messages=[],
            user_query=user_input,
            rag_context=[],
            response="",
            session_id=session_id,
            token_count=0,
        )

        try:
            result = self.graph.invoke(initial_state, config)
            raw_response = result.get("response", "")
            cleaned_response = self.clean_response(raw_response)

            metadata = {
                "session_id": session_id,
                "token_count": int(result.get("token_count", 0)),
                "rag_chunks_used": len(result.get("rag_context", [])),
                "conversation_length": len(result.get("messages", [])),
                "timestamp": datetime.now().isoformat(),
                "ollama_available": self.rag_engine.ollama_available,
            }

            logger.info(f"Chat completed for session {session_id}: {metadata}")
            return cleaned_response, session_id, metadata

        except Exception as exc:
            logger.error(f"Error in chat processing: {exc}")
            return f"I encountered an error: {exc}", session_id, {"error": str(exc)}

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get the conversation history for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of conversation messages.
        """
        try:
            return self.conversation_memory.get_recent_messages(session_id, limit=100)
        except Exception as exc:
            logger.error(f"Error getting conversation history: {exc}")
            return []

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session."""
        try:
            total_tokens = self.conversation_memory.get_total_tokens(session_id)
            messages = self.conversation_memory.get_recent_messages(
                session_id, limit=1000
            )

            return {
                "session_id": session_id,
                "total_messages": len(messages),
                "total_tokens": total_tokens,
                "user_messages": len([m for m in messages if m["type"] == "user"]),
                "assistant_messages": len(
                    [m for m in messages if m["type"] == "assistant"]
                ),
            }
        except Exception as exc:
            logger.error(f"Error getting session stats: {exc}")
            return {}

    def clear_session_memory(self, session_id: str) -> bool:
        """Clear all memory for a specific session."""
        try:
            self.conversation_memory.clear_session(session_id)
            logger.info(f"Cleared memory for session: {session_id}")
            return True
        except Exception as exc:
            logger.error(f"Error clearing session memory: {exc}")
            return False

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        return self.rag_engine.get_database_stats()

    def close(self) -> None:
        """Close the conversation engine and its resources."""
        self.rag_engine.close()
        logger.info("Conversation engine closed")
