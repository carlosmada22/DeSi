#!/usr/bin/env python3
"""
RAG Query Engine for DeSi

This module provides a RAG (Retrieval Augmented Generation) query engine
that uses ChromaDB vector database for efficient similarity search across
multiple documentation sources (ReadTheDocs and Wiki.js).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama, but don't fail if it's not available
try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    OLLAMA_AVAILABLE = True
    # Test if Ollama server is running
    try:
        test_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embedding = test_embeddings.embed_query("test")
        if not test_embedding:
            OLLAMA_AVAILABLE = False
    except Exception:
        OLLAMA_AVAILABLE = False
except ImportError:
    logger.warning("Langchain Ollama package not available.")
    OLLAMA_AVAILABLE = False

# Import vector database
from ..utils.vector_db import DesiVectorDB


class DesiRAGQueryEngine:
    """RAG Query Engine for DeSi using ChromaDB vector database."""

    def __init__(
        self,
        db_path: str = "desi_vectordb",
        collection_name: str = "desi_docs",
        model: str = "gpt-oss:20b"
    ):
        """
        Initialize the RAG query engine.

        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the ChromaDB collection
            model: The Ollama model to use for chat
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model

        # Initialize vector database
        self.vector_db = DesiVectorDB(
            db_path=db_path,
            collection_name=collection_name
        )

        if OLLAMA_AVAILABLE:
            logger.info("Using Ollama for embeddings and completions")
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
            self.llm = ChatOllama(model=self.model)
        else:
            logger.warning("Ollama not available or not running")
            self.embeddings_model = None
            self.llm = None

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding for the text
        """
        if OLLAMA_AVAILABLE and self.embeddings_model:
            try:
                # Use Ollama's embedding API
                embedding = self.embeddings_model.embed_query(text)
                return embedding

            except Exception as e:
                logger.error(f"Error generating embedding with Ollama: {e}")
                logger.warning("Falling back to dummy embedding")

        # If Ollama is not available or there was an error, use a dummy embedding
        return self._generate_dummy_embedding()

    def _generate_dummy_embedding(self, dim: int = 1536) -> List[float]:
        """
        Generate a dummy embedding (random vector).

        Args:
            dim: The dimension of the embedding

        Returns:
            A random vector of the specified dimension
        """
        # Generate a random vector
        embedding = np.random.normal(0, 1, dim)

        # Normalize it to unit length
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def retrieve_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 5,
        prioritize_datastore: bool = True
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query using vector similarity search.

        Args:
            query: The query to retrieve chunks for
            top_k: The number of chunks to retrieve
            prioritize_datastore: Whether to prioritize datastore (Wiki.js) content

        Returns:
            A list of relevant chunks with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)

            # Determine source boosting strategy
            source_boost = None
            if prioritize_datastore:
                source_boost = "datastore"  # Boost Wiki.js content

            # Search the vector database
            results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=top_k,
                source_boost=source_boost
            )

            logger.info(f"Retrieved {len(results)} relevant chunks for query: {query}")
            
            # Log source distribution
            sources = {}
            for result in results:
                source = result.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            logger.info(f"Source distribution in results: {sources}")

            return results

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []

    def _create_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Create a prompt for the language model using the query and relevant chunks.

        Args:
            query: The user's query
            relevant_chunks: The relevant chunks to include in the prompt

        Returns:
            The formatted prompt
        """
        # Build the context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.get('source', 'unknown')
            title = chunk.get('title', 'Unknown Document')
            content = chunk.get('content', '')
            
            # Add source indicator
            source_label = "openBIS Documentation" if source == "openbis" else "Data Store Wiki"
            
            context_parts.append(f"[Context {i} - {source_label}]\nTitle: {title}\nContent: {content}\n")

        context = "\n".join(context_parts)

        # Create the prompt
        prompt = f"""Based on the following context from openBIS documentation and Data Store wiki, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a helpful, accurate answer based on the context provided. If the context contains information from both openBIS documentation and the Data Store wiki, prioritize the Data Store wiki information when relevant, as it may contain more specific or up-to-date information for data store operations.

Answer:"""

        return prompt

    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Generate an answer for a query using the relevant chunks.

        Args:
            query: The query to generate an answer for
            relevant_chunks: The relevant chunks to use for generating the answer

        Returns:
            The generated answer
        """
        if not OLLAMA_AVAILABLE or not self.llm:
            return "Ollama not available or not running. Cannot generate answer."

        try:
            # Create a prompt for the language model
            prompt = self._create_prompt(query, relevant_chunks)

            # Create the system instruction
            system_instruction = """You are DeSi, a knowledgeable assistant specializing in openBIS and data store operations.
You provide friendly, clear, and accurate answers based on documentation from both openBIS and the Data Store wiki.

Guidelines:
- Be helpful and conversational
- Provide step-by-step instructions when appropriate
- When information is available from both sources, prioritize Data Store wiki content for data store-specific operations
- If you're not sure about something, say so rather than guessing
- Use examples when they help clarify your answer

<think>
Let me analyze the context and query to provide the most helpful answer.
</think>"""

            # Generate the response using Ollama
            full_prompt = system_instruction + "\n\n" + prompt
            response = self.llm.invoke(full_prompt)

            # Extract the answer from the response
            answer = response.content

            # Store the original answer for debugging
            self.original_answer = answer

            # Remove the <think>...</think> tags and their contents
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

            # Trim any extra whitespace that might be left
            answer = answer.strip()

            # Log the original answer for debugging
            logger.debug(f"Original answer with thinking: {self.original_answer}")

            return answer

        except Exception as e:
            logger.error(f"Error generating answer with Ollama: {e}")
            return f"Error generating answer: {e}"

    def query(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Query the vector database using RAG.

        Args:
            query: The query to answer
            top_k: The number of chunks to retrieve

        Returns:
            A tuple containing the answer and the relevant chunks
        """
        # Retrieve the most relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)

        # Generate an answer
        answer = self.generate_answer(query, relevant_chunks)

        # Store the original answer with thinking in the metadata
        # This can be used for debugging or analysis later
        metadata = {"original_query": query}
        if hasattr(self, "original_answer"):
            metadata["original_answer"] = self.original_answer

        return answer, relevant_chunks

    def get_database_stats(self) -> Dict:
        """
        Get statistics about the vector database.

        Returns:
            Dictionary with database statistics
        """
        return self.vector_db.get_collection_stats()

    def close(self) -> None:
        """
        Close the vector database connection.
        """
        self.vector_db.close()
