#!/usr/bin/env python3
"""
Vector Database utilities for DeSi using ChromaDB.

Provides a thin wrapper around ChromaDB with a graceful in-memory fallback when
Chroma is not available (e.g. in unit-test environments).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly in environments with ChromaDB
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path for tests without chromadb
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    CHROMADB_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class DesiVectorDB:
    """Vector database manager for DeSi using ChromaDB or an in-memory fallback."""

    def __init__(
        self, db_path: str = "desi_vectordb", collection_name: str = "desi_docs"
    ):
        """
        Initialize the vector database.

        Args:
            db_path: Path to the ChromaDB database directory.
            collection_name: Name of the collection to use.
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Ensure the backing directory exists (handy for the fallback as well)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self._in_memory = not CHROMADB_AVAILABLE
        if self._in_memory:
            self._records: List[Dict[str, Any]] = []
            logger.warning(
                "ChromaDB not available. Using in-memory vector store for DesiVectorDB."
            )
            return

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(  # type: ignore[union-attr]
            path=str(self.db_path),
            settings=Settings(  # type: ignore[operator]
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing collection '{collection_name}'")
        except Exception:
            logger.info(f"Collection '{collection_name}' doesn't exist, creating it...")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "DeSi Helper documentation chunks with embeddings"
                    },
                )
                logger.info(f"Created new collection '{collection_name}'")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                raise

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Add chunks to the vector database."""
        if not chunks:
            logger.warning("No chunks to add to vector database")
            return

        if self._in_memory:
            records_added = 0
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id") or f"chunk_{len(self._records)}"
                embedding = chunk.get("embedding")
                if not embedding:
                    logger.warning(f"Chunk {chunk_id} has no embedding, skipping")
                    continue

                emb_array = np.asarray(embedding, dtype=float)
                if emb_array.size == 0:
                    continue

                norm = float(np.linalg.norm(emb_array))
                metadata = {
                    k: v
                    for k, v in chunk.items()
                    if k not in {"embedding", "content"} and v is not None
                }

                # Replace existing record with same id
                self._records = [
                    record for record in self._records if record["id"] != chunk_id
                ]
                self._records.append(
                    {
                        "id": chunk_id,
                        "embedding": emb_array,
                        "norm": norm,
                        "document": chunk.get("content", ""),
                        "metadata": metadata,
                    }
                )
                records_added += 1

            logger.info(f"Added {records_added} chunks to in-memory vector store")
            return

        # Prepare data for ChromaDB
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", f"chunk_{len(ids)}")
            embedding = chunk.get("embedding", [])
            if not embedding:
                logger.warning(f"Chunk {chunk_id} has no embedding, skipping")
                continue

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.get("content", ""))
            metadata = {
                k: v
                for k, v in chunk.items()
                if k not in ["embedding", "content"] and v is not None
            }
            metadatas.append(metadata)

        if not embeddings:
            logger.warning("No valid embeddings found in chunks")
            return

        try:
            self.collection.add(  # type: ignore[union-attr]
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(embeddings)} chunks to vector database")
        except Exception as exc:
            logger.error(f"Error adding chunks to vector database: {exc}")
            raise

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        source_filter: Optional[str] = None,
        source_boost: Optional[str] = None,
    ) -> List[Dict]:
        """Search for similar chunks in the vector database."""
        if self._in_memory:
            if not query_embedding or not self._records:
                return []

            query_vec = np.asarray(query_embedding, dtype=float)
            if query_vec.size == 0:
                return []

            query_norm = float(np.linalg.norm(query_vec))
            if query_norm == 0:
                return []

            def cosine_similarity(record: Dict[str, Any]) -> float:
                denom = query_norm * record.get("norm", 0.0)
                if denom == 0:
                    return 0.0
                return float(np.dot(query_vec, record["embedding"]) / denom)

            records = self._records
            if source_filter:
                records = [
                    record
                    for record in records
                    if record["metadata"].get("source") == source_filter
                ]

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for record in records:
                score = cosine_similarity(record)
                if (
                    source_boost
                    and not source_filter
                    and record["metadata"].get("source") == source_boost
                ):
                    score += 0.05  # light boost
                scored.append((score, record))

            scored.sort(key=lambda item: item[0], reverse=True)
            top_records = scored[:n_results]

            formatted_results = []
            for score, record in top_records:
                chunk = {
                    "chunk_id": record["id"],
                    "content": record["document"],
                    "similarity_score": score,
                    **record["metadata"],
                }
                formatted_results.append(chunk)

            logger.info(f"Found {len(formatted_results)} similar chunks (in-memory)")
            return formatted_results

        try:
            where_clause = {"source": source_filter} if source_filter else None
            logger.debug(
                f"ChromaDB search: where_clause={where_clause}, n_results={n_results}"
            )

            if source_boost and not source_filter:
                boosted_results = self.collection.query(  # type: ignore[union-attr]
                    query_embeddings=[query_embedding],
                    n_results=max(1, n_results // 2),
                    where={"source": source_boost},
                )
                remaining_results = self.collection.query(  # type: ignore[union-attr]
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=None,
                )

                all_ids = set()
                combined_results = {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                }

                for i, doc_id in enumerate(boosted_results["ids"][0]):
                    if doc_id not in all_ids:
                        all_ids.add(doc_id)
                        combined_results["ids"][0].append(doc_id)
                        combined_results["distances"][0].append(
                            boosted_results["distances"][0][i]
                        )
                        combined_results["documents"][0].append(
                            boosted_results["documents"][0][i]
                        )
                        combined_results["metadatas"][0].append(
                            boosted_results["metadatas"][0][i]
                        )

                for i, doc_id in enumerate(remaining_results["ids"][0]):
                    if (
                        doc_id not in all_ids
                        and len(combined_results["ids"][0]) < n_results
                    ):
                        all_ids.add(doc_id)
                        combined_results["ids"][0].append(doc_id)
                        combined_results["distances"][0].append(
                            remaining_results["distances"][0][i]
                        )
                        combined_results["documents"][0].append(
                            remaining_results["documents"][0][i]
                        )
                        combined_results["metadatas"][0].append(
                            remaining_results["metadatas"][0][i]
                        )

                results = combined_results
            else:
                results = self.collection.query(  # type: ignore[union-attr]
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                )

            formatted_results = []
            for i in range(len(results["ids"][0])):
                chunk = {
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],
                    **results["metadatas"][0][i],
                }
                formatted_results.append(chunk)

            logger.info(f"Found {len(formatted_results)} similar chunks")
            if formatted_results:
                logger.debug(
                    f"Sample result: {formatted_results[0].get('chunk_id')} (score: {formatted_results[0].get('similarity_score'):.4f})"
                )
            return formatted_results

        except Exception as exc:
            logger.error(f"Error searching vector database: {exc}")
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if self._in_memory:
            sources: Dict[str, int] = {}
            for record in self._records:
                source = record["metadata"].get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            return {
                "total_chunks": len(self._records),
                "collection_name": self.collection_name,
                "source_distribution": sources,
            }

        try:
            count = self.collection.count()  # type: ignore[union-attr]
            sample_results = self.collection.get(limit=min(100, count))  # type: ignore[union-attr]
            sources: Dict[str, int] = {}
            for metadata in sample_results.get("metadatas", []):
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "source_distribution": sources,
            }
        except Exception as exc:
            logger.error(f"Error getting collection stats: {exc}")
            return {}

    def reset_collection(self) -> None:
        """Reset the collection (delete all data)."""
        if self._in_memory:
            self._records.clear()
            logger.info("Reset in-memory vector store")
            return

        try:
            self.client.delete_collection(name=self.collection_name)  # type: ignore[union-attr]
            self.collection = self.client.create_collection(  # type: ignore[union-attr]
                name=self.collection_name,
                metadata={"description": "DeSi documentation chunks with embeddings"},
            )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as exc:
            logger.error(f"Error resetting collection: {exc}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        logger.info(
            "Vector database connection closed%s",
            " (in-memory)" if self._in_memory else "",
        )
