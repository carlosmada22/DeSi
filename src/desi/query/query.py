#!/usr/bin/env python3
"""
RAG Query Engine for DeSi

This module coordinates query embedding, similarity search, and prompt construction
on top of the ChromaDB vector database produced by the unified document processor.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.vector_db import DesiVectorDB

logger = logging.getLogger(__name__)

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings  # type: ignore

    _OLLAMA_MODULE_AVAILABLE = True
except ImportError:
    ChatOllama = None  # type: ignore
    OllamaEmbeddings = None  # type: ignore
    _OLLAMA_MODULE_AVAILABLE = False
    logger.warning("Langchain Ollama package not available.")

DATASTORE_SOURCES = {"datastore", "wikijs"}
DEFAULT_TOP_K = 5
DEFAULT_EMBED_DIM = 768


class DesiRAGQueryEngine:
    """RAG Query Engine for DeSi using ChromaDB vector database."""

    def __init__(
        self,
        db_path: str = "desi_vectordb",
        collection_name: str = "desi_docs",
        model: str = "gpt-oss:20b",
    ):
        """
        Initialize the RAG query engine.

        Args:
            db_path: Path to the ChromaDB database directory.
            collection_name: Name of the ChromaDB collection.
            model: The Ollama model to use for chat.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model

        self.vector_db = DesiVectorDB(
            db_path=db_path,
            collection_name=collection_name,
        )

        self.embedding_dimension = DEFAULT_EMBED_DIM
        self.embeddings_model = None
        self.llm = None
        self.ollama_available = False
        self.original_answer: Optional[str] = None
        self.last_prompt: Optional[str] = None
        self.last_retrieved_chunks: List[Dict] = []

        self._configure_embedding_dimension()
        self._configure_ollama_clients()

    def _configure_embedding_dimension(self) -> None:
        """Infer embedding dimension from the existing collection when possible."""
        try:
            sample = self.vector_db.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings and embeddings[0]:
                self.embedding_dimension = len(embeddings[0])
                logger.debug(
                    "Detected embedding dimension from collection: %s",
                    self.embedding_dimension,
                )
        except Exception as exc:
            logger.debug("Could not infer embedding dimension: %s", exc)

    def _configure_ollama_clients(self) -> None:
        """Initialise Ollama embedding and chat clients if available."""
        if not _OLLAMA_MODULE_AVAILABLE:
            logger.info("Ollama integrations disabled; running in fallback mode.")
            return

        try:
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")  # type: ignore[operator]
            probe_vector = self.embeddings_model.embed_query("dimension probe")  # type: ignore[union-attr]
            if probe_vector:
                self.embedding_dimension = len(probe_vector)
            self.llm = ChatOllama(model=self.model)  # type: ignore[operator]
            self.ollama_available = True
            logger.info(
                "Using Ollama for embeddings and completions (dimension=%s).",
                self.embedding_dimension,
            )
        except Exception as exc:
            logger.warning("Ollama not available or not running: %s", exc)
            self.embeddings_model = None
            self.llm = None
            self.ollama_available = False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a piece of text."""
        if self.embeddings_model:
            try:
                embedding = self.embeddings_model.embed_query(text)  # type: ignore[union-attr]
                if embedding:
                    self.embedding_dimension = len(embedding)
                return embedding
            except Exception as exc:
                logger.error("Error generating embedding with Ollama: %s", exc)
                logger.warning("Falling back to dummy embedding.")

        return self._generate_dummy_embedding()

    def _generate_dummy_embedding(self, dim: Optional[int] = None) -> List[float]:
        """Generate a dummy embedding (normalised random vector)."""
        dimension = dim or self.embedding_dimension or DEFAULT_EMBED_DIM
        vector = np.random.normal(0, 1, dimension)
        norm = np.linalg.norm(vector)
        if not norm:
            return [0.0] * dimension
        return (vector / norm).tolist()

    def _normalize_chunk(self, raw_chunk: Dict) -> Dict:
        """Normalise raw search results into a consistent structure."""
        chunk = dict(raw_chunk) if raw_chunk else {}

        document_id = str(
            chunk.get("chunk_id")
            or chunk.get("id")
            or chunk.get("document_id")
            or chunk.get("path")
            or chunk.get("title")
            or f"chunk-{len(self.last_retrieved_chunks)}"
        )

        source = (chunk.get("source") or chunk.get("repo") or "unknown").lower()
        title = (
            chunk.get("title")
            or chunk.get("chunk_title")
            or chunk.get("section")
            or "Untitled"
        )
        section = chunk.get("section") or chunk.get("section_title") or ""
        path = chunk.get("path") or chunk.get("file_path") or ""
        content = (
            chunk.get("content") or chunk.get("document") or chunk.get("text") or ""
        )

        try:
            similarity = float(chunk.get("similarity_score", 0.0))
        except (TypeError, ValueError):
            similarity = 0.0

        try:
            priority = int(chunk.get("source_priority", 99))
        except (TypeError, ValueError):
            priority = 99

        source_label = (
            "Data Store Wiki"
            if source in DATASTORE_SOURCES
            else "openBIS Documentation"
        )

        metadata = {
            k: v for k, v in chunk.items() if k not in {"content", "similarity_score"}
        }
        metadata.setdefault("source", source)
        metadata.setdefault("title", title)
        metadata.setdefault("section", section)
        metadata.setdefault("path", path)

        return {
            "document_id": document_id,
            "chunk_id": chunk.get("chunk_id") or document_id,
            "source": source,
            "source_label": source_label,
            "title": title,
            "section": section,
            "path": path,
            "content": content,
            "similarity_score": similarity,
            "source_priority": priority,
            "metadata": metadata,
        }

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        prioritize_datastore: bool = True,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query using vector similarity search.

        Args:
            query: The query to retrieve chunks for.
            top_k: The number of chunks to return.
            prioritize_datastore: Whether to prioritise datastore (Wiki.js) content.

        Returns:
            A list of relevant chunks with normalised metadata.
        """
        if not query or not query.strip():
            logger.debug("Empty query received; returning no chunks.")
            return []

        candidates: Dict[str, Dict] = {}
        query_embedding = self.generate_embedding(query)
        search_limit = max(top_k * 2, top_k + 2)

        def collect(results: List[Dict]) -> None:
            logger.debug(f"Collecting {len(results)} raw results")
            for raw in results:
                normalised = self._normalize_chunk(raw)
                doc_id = normalised["document_id"]
                existing = candidates.get(doc_id)
                if existing:
                    if normalised["similarity_score"] > existing["similarity_score"]:
                        existing["similarity_score"] = normalised["similarity_score"]
                        existing["metadata"] = normalised["metadata"]
                else:
                    candidates[doc_id] = normalised

        try:
            logger.debug(f"Starting search for query: '{query}'")
            base_results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=search_limit,
            )
            logger.debug(f"Base search returned {len(base_results)} results")
            collect(base_results)

            if prioritize_datastore:
                for source_name in DATASTORE_SOURCES:
                    logger.debug(f"Searching with source filter: {source_name}")
                    boosted = self.vector_db.search(
                        query_embedding=query_embedding,
                        n_results=top_k,
                        source_filter=source_name,
                    )
                    logger.debug(
                        f"Source-filtered search for '{source_name}' returned {len(boosted)} results"
                    )
                    collect(boosted)

            logger.debug(f"Total candidates collected: {len(candidates)}")
            ranked = sorted(
                candidates.values(),
                key=lambda item: (item["source_priority"], -item["similarity_score"]),
            )

            top_chunks = ranked[:top_k]
            self.last_retrieved_chunks = top_chunks

            distribution: Dict[str, int] = {}
            for chunk in top_chunks:
                distribution[chunk["source"]] = distribution.get(chunk["source"], 0) + 1

            logger.info(
                "Retrieved %d chunks for query '%s' (distribution: %s)",
                len(top_chunks),
                query,
                distribution,
            )
            return top_chunks
        except Exception as exc:
            logger.error("Error retrieving relevant chunks: %s", exc)
            return []

    def _create_prompt(
        self,
        query: str,
        relevant_chunks: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Create a prompt for the language model using the query, context, and history."""
        history_block = ""
        if conversation_history:
            formatted_history = []
            for item in conversation_history[-6:]:
                role = (item.get("role") or item.get("type") or "user").lower()
                label = "User" if role == "user" else "Assistant"
                content = item.get("content", "").strip()
                if content:
                    formatted_history.append(f"{label}: {content}")
            if formatted_history:
                history_block = (
                    "Previous conversation:\n" + "\n".join(formatted_history) + "\n\n"
                )

        if relevant_chunks:
            context_parts = []
            for idx, chunk in enumerate(relevant_chunks, start=1):
                lines = [
                    f"[Context {idx} - {chunk['source_label']}]",
                    f"Title: {chunk['title']}",
                ]
                if chunk.get("section"):
                    lines.append(f"Section: {chunk['section']}")
                if chunk.get("path"):
                    lines.append(f"Path: {chunk['path']}")
                lines.append("Content:")
                lines.append(chunk.get("content", ""))
                context_parts.append("\n".join(lines))
            context_block = "\n\n".join(context_parts)
        else:
            context_block = "No documentation snippets were retrieved."

        if relevant_chunks:
            prompt = (
                f"{history_block}"
                "Use the documentation context below to answer the user's question.\n"
                "Documentation Context:\n"
                f"{context_block}\n\n"
                f"User Question: {query}\n\n"
                "Instructions:\n"
                "- Provide clear, actionable guidance grounded in the context.\n"
                "- Cite relevant sections briefly when helpful.\n"
                "- Prioritise details from the Data Store wiki when available.\n"
                "- Only use information from the provided context.\n"
                "- If the context doesn't contain enough information, say so clearly.\n\n"
                "Answer:"
            )
        else:
            prompt = (
                f"{history_block}"
                f"User Question: {query}\n\n"
                "I don't have access to relevant documentation for this question. "
                "The vector database search didn't return any matching content. "
                "Please try rephrasing your question or contact the system administrator "
                "if you believe this information should be available.\n\n"
                "Answer: I don't have information about that topic in my current knowledge base."
            )
        return prompt

    @staticmethod
    def _strip_think_tags(answer: str) -> str:
        """Remove <think> tags from Ollama responses."""
        if not answer:
            return ""
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    def _render_fallback_answer(
        self,
        query: str,
        relevant_chunks: List[Dict],
        error: Optional[str] = None,
    ) -> str:
        """Fallback response when Ollama is unavailable."""
        if not relevant_chunks:
            base = "I could not find relevant documentation snippets to answer that question."
        else:
            bullet_lines = []
            for chunk in relevant_chunks:
                snippet_lines = [
                    line.strip()
                    for line in chunk.get("content", "").splitlines()
                    if line.strip()
                ]
                snippet = " ".join(snippet_lines)[:280]
                bullet_lines.append(
                    f"- {chunk['source_label']} · {chunk['title']}: {snippet}"
                )
            base = (
                "Ollama is unavailable, so here is a summary of the retrieved documentation:\n"
                + "\n".join(bullet_lines)
            )

        if error:
            base += f"\n\n(Generation fallback triggered due to: {error})"
        return base

    def generate_answer(
        self,
        query: str,
        relevant_chunks: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate an answer for a query using the relevant chunks.

        Args:
            query: The query to answer.
            relevant_chunks: Retrieved context chunks.
            conversation_history: Optional conversation history for additional context.

        Returns:
            The generated answer.
        """
        if not self.ollama_available or not self.llm:
            return self._render_fallback_answer(query, relevant_chunks)

        try:
            prompt = self._create_prompt(query, relevant_chunks, conversation_history)
            self.last_prompt = prompt

            system_instruction = (
                "You are DeSi, a knowledgeable assistant specialising in openBIS and data store operations.\n"
                "IMPORTANT: Only use information from the provided documentation context. "
                "Do not make up or invent information. If the context doesn't contain "
                "the answer, clearly state that you don't have that information.\n"
                "Respond in clear, friendly language and ground each answer in the provided documentation.\n"
                "Prioritise Data Store wiki information when it is present."
            )

            full_prompt = system_instruction + "\n\n" + prompt
            response = self.llm.invoke(full_prompt)  # type: ignore[union-attr]

            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            self.original_answer = answer_text

            cleaned = self._strip_think_tags(answer_text)
            return cleaned
        except Exception as exc:
            logger.error("Error generating answer with Ollama: %s", exc)
            return self._render_fallback_answer(query, relevant_chunks, error=str(exc))

    def query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Query the vector database using RAG.

        Args:
            query: The query to answer.
            top_k: The number of chunks to retrieve.
            conversation_history: Optional conversation history to include.

        Returns:
            A tuple containing the answer and the relevant chunks.
        """
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)
        answer = self.generate_answer(
            query,
            relevant_chunks,
            conversation_history=conversation_history,
        )
        return answer, relevant_chunks

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        return self.vector_db.get_collection_stats()

    def close(self) -> None:
        """Close the vector database connection."""
        self.vector_db.close()
