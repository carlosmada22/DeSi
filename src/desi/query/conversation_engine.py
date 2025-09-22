#!/usr/bin/env python3
"""
Simplified Conversation Engine for DeSi

This module provides a simplified conversation engine that maintains memory
across multiple interactions using LangGraph's state management, but focuses
exclusively on RAG functionality without function calling or routing.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Annotated
from datetime import datetime
import uuid
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .query import DesiRAGQueryEngine

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Langchain Ollama package not available.")
    OLLAMA_AVAILABLE = False


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

    def add_message(self, session_id: str, message_type: str, content: str,
                   token_count: int = 0, metadata: Dict = None):
        """Add a message to the conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute("""
                INSERT INTO conversations
                (session_id, message_type, content, token_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, message_type, content, token_count, metadata_json))
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get the most recent messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_type, content, token_count, timestamp, metadata
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))

            messages = []
            for row in reversed(cursor.fetchall()):  # Reverse to get chronological order
                message_type, content, token_count, timestamp, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                messages.append({
                    'type': message_type,
                    'content': content,
                    'token_count': token_count,
                    'timestamp': timestamp,
                    'metadata': metadata
                })
            return messages

    def get_total_tokens(self, session_id: str) -> int:
        """Get total token count for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(token_count) FROM conversations WHERE session_id = ?
            """, (session_id,))
            result = cursor.fetchone()[0]
            return result if result else 0

    def clear_session(self, session_id: str):
        """Clear all messages for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()


class ConversationState(TypedDict):
    """State for the conversation graph."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    rag_context: List[Dict]
    response: str
    session_id: str
    token_count: int


class DesiConversationEngine:
    """Simplified conversation engine for DeSi with memory and RAG integration."""

    def __init__(
        self,
        db_path: str = "desi_vectordb",
        collection_name: str = "desi_docs",
        model: str = "gpt-oss:20b",
        memory_db_path: str = "desi_conversation_memory.db"
    ):
        """
        Initialize the conversation engine.

        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the ChromaDB collection
            model: Ollama model to use
            memory_db_path: Path to SQLite database for conversation memory
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.memory_db_path = memory_db_path

        # Initialize RAG engine
        self.rag_engine = DesiRAGQueryEngine(
            db_path=db_path,
            collection_name=collection_name,
            model=model
        )

        # Initialize LLM for conversation
        if OLLAMA_AVAILABLE:
            self.llm = ChatOllama(model=model)
        else:
            self.llm = None
            logger.warning("Ollama not available. Conversation features will be limited.")

        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(memory_db_path)

        # Initialize LangGraph memory (for state management)
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{memory_db_path}")

        # Build the conversation graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation flow."""

        def rag_agent(state: ConversationState) -> ConversationState:
            """RAG agent for documentation queries."""
            try:
                # Get relevant chunks for the current query
                relevant_chunks = self.rag_engine.retrieve_relevant_chunks(
                    state["user_query"], top_k=3
                )
                state["rag_context"] = relevant_chunks
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

                # Generate RAG response
                if not OLLAMA_AVAILABLE or not self.llm:
                    state["response"] = "Ollama not available. Cannot generate response."
                    return state

                # Build conversation context
                messages = []

                # Add system message
                system_msg = SystemMessage(content="""You are DeSi, a knowledgeable assistant specializing in openBIS and data store operations.
You provide friendly, clear, and accurate answers based on documentation from both openBIS and the Data Store wiki.

Guidelines:
- Be helpful and conversational
- Provide step-by-step instructions when appropriate
- When information is available from both sources, prioritize Data Store wiki content for data store-specific operations
- If you're not sure about something, say so rather than guessing
- Use examples when they help clarify your answer
- Keep track of the conversation context and refer to previous messages when relevant""")
                messages.append(system_msg)

                # Load conversation history from SQLite memory (last 20 messages)
                session_id = state.get("session_id", "default")
                recent_memory = self.conversation_memory.get_recent_messages(session_id, limit=20)

                # Convert memory to LangChain messages
                for msg in recent_memory:
                    if msg['type'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg['type'] == 'assistant':
                        messages.append(AIMessage(content=msg['content']))

                # Add context from RAG
                if relevant_chunks:
                    context_parts = []
                    for i, chunk in enumerate(relevant_chunks, 1):
                        source = chunk.get('source', 'unknown')
                        title = chunk.get('title', 'Unknown Document')
                        content = chunk.get('content', '')
                        
                        # Add source indicator
                        source_label = "openBIS Documentation" if source == "openbis" else "Data Store Wiki"
                        
                        context_parts.append(f"[Context {i} - {source_label}]\nTitle: {title}\nContent: {content}\n")

                    context = "\n".join(context_parts)
                    context_msg = SystemMessage(content=f"Relevant documentation context:\n{context}")
                    messages.append(context_msg)

                # Add the current user message
                messages.append(HumanMessage(content=state["user_query"]))

                # Generate response
                response = self.llm.invoke(messages)
                state["response"] = response.content

                # Estimate token count (rough approximation)
                total_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
                state["token_count"] = len(total_text.split()) * 1.3  # Rough token estimation

                logger.info(f"Generated RAG response with estimated {state['token_count']} tokens")
                return state

            except Exception as e:
                logger.error(f"Error in RAG agent: {e}")
                state["response"] = f"I encountered an error: {str(e)}"
                state["rag_context"] = []
                return state

        def update_conversation(state: ConversationState) -> ConversationState:
            """Update the conversation history with the new exchange."""
            try:
                session_id = state.get("session_id", "default")

                # Calculate token counts for user and assistant messages
                user_tokens = int(len(state["user_query"].split()) * 1.3)
                assistant_tokens = int(len(state["response"].split()) * 1.3)

                # Save user message to SQLite memory
                self.conversation_memory.add_message(
                    session_id=session_id,
                    message_type="user",
                    content=state["user_query"],
                    token_count=user_tokens,
                    metadata={"rag_context_count": len(state.get("rag_context", []))}
                )

                # Save assistant response to SQLite memory
                self.conversation_memory.add_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=state["response"],
                    token_count=assistant_tokens,
                    metadata={"model": self.model}
                )

                # Update total token count
                total_tokens = self.conversation_memory.get_total_tokens(session_id)
                state["token_count"] = total_tokens

                logger.info(f"Saved conversation to memory. Session tokens: {total_tokens}")
                return state
                
            except Exception as e:
                logger.error(f"Error updating conversation: {e}")
                return state

        # Build the graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("rag_agent", rag_agent)
        workflow.add_node("update_conversation", update_conversation)

        # Set entry point
        workflow.set_entry_point("rag_agent")

        # Add edges
        workflow.add_edge("rag_agent", "update_conversation")
        workflow.add_edge("update_conversation", END)

        # Compile the graph with memory
        return workflow.compile(checkpointer=self.memory)

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id

    def clean_response(self, response: str) -> str:
        """Remove <think></think> tags from the response."""
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def chat(self, user_input: str, session_id: Optional[str] = None) -> Tuple[str, str, Dict]:
        """
        Process a user input and return the response.

        Args:
            user_input: The user's message
            session_id: Optional session ID for conversation continuity

        Returns:
            Tuple of (response, session_id, metadata)
        """
        if not session_id:
            session_id = self.create_session()

        # Create config for this conversation thread
        config = RunnableConfig(
            configurable={"thread_id": session_id}
        )

        # Create initial state
        initial_state = ConversationState(
            messages=[],
            user_query=user_input,
            rag_context=[],
            response="",
            session_id=session_id,
            token_count=0
        )

        try:
            # Run the conversation graph
            result = self.graph.invoke(initial_state, config)

            # Extract and clean response
            raw_response = result["response"]
            cleaned_response = self.clean_response(raw_response)

            metadata = {
                "session_id": session_id,
                "token_count": result["token_count"],
                "rag_chunks_used": len(result["rag_context"]),
                "conversation_length": len(result["messages"]),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Chat completed for session {session_id}: {metadata}")
            return cleaned_response, session_id, metadata

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I encountered an error: {str(e)}", session_id, {"error": str(e)}

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get the conversation history for a session.

        Args:
            session_id: The session ID

        Returns:
            List of conversation messages
        """
        try:
            # Get conversation history from SQLite memory
            return self.conversation_memory.get_recent_messages(session_id, limit=100)
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session."""
        try:
            total_tokens = self.conversation_memory.get_total_tokens(session_id)
            messages = self.conversation_memory.get_recent_messages(session_id, limit=1000)

            return {
                "session_id": session_id,
                "total_messages": len(messages),
                "total_tokens": total_tokens,
                "user_messages": len([m for m in messages if m['type'] == 'user']),
                "assistant_messages": len([m for m in messages if m['type'] == 'assistant'])
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}

    def clear_session_memory(self, session_id: str) -> bool:
        """Clear all memory for a specific session."""
        try:
            self.conversation_memory.clear_session(session_id)
            logger.info(f"Cleared memory for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session memory: {e}")
            return False

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        return self.rag_engine.get_database_stats()

    def close(self) -> None:
        """Close the conversation engine and its resources."""
        self.rag_engine.close()
        logger.info("Conversation engine closed")
