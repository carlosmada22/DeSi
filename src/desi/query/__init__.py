"""
Query and Conversation module for DeSi.

This module contains the RAG query engine, a stateful conversation memory manager,
and a LangGraph-based chatbot engine to orchestrate the entire conversational
RAG pipeline.
"""

# Import the main classes to make them accessible at the package level
from .conversation_engine import ChatbotEngine, SqliteConversationMemory
from .query import RAGQueryEngine

# Define the public API of this package
__all__ = [
    "RAGQueryEngine",
    "ChatbotEngine",
    "SqliteConversationMemory",
]
