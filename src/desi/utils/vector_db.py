#!/usr/bin/env python3
"""
Vector Database utilities for DeSi using ChromaDB.

This module provides functionality to store and retrieve embeddings
using ChromaDB as the vector database backend.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import chromadb
from chromadb.config import Settings

# Configure logging
logger = logging.getLogger(__name__)


class DesiVectorDB:
    """Vector database manager for DeSi using ChromaDB."""

    def __init__(self, db_path: str = "desi_vectordb", collection_name: str = "desi_docs"):
        """
        Initialize the vector database.

        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the collection to use
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Create the database directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing collection '{collection_name}'")
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection '{collection_name}' doesn't exist, creating it...")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "DeSi Helper documentation chunks with embeddings"}
                )
                logger.info(f"Created new collection '{collection_name}'")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                raise

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the vector database.

        Args:
            chunks: List of chunk dictionaries with embeddings
        """
        if not chunks:
            logger.warning("No chunks to add to vector database")
            return

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Use chunk_id as the unique identifier
            chunk_id = chunk.get("chunk_id", f"chunk_{len(ids)}")
            ids.append(chunk_id)
            
            # Extract embedding
            embedding = chunk.get("embedding", [])
            if not embedding:
                logger.warning(f"Chunk {chunk_id} has no embedding, skipping")
                continue
            embeddings.append(embedding)
            
            # Use content as the document text
            documents.append(chunk.get("content", ""))
            
            # Store metadata (everything except embedding and content)
            metadata = {k: v for k, v in chunk.items() 
                       if k not in ["embedding", "content"] and v is not None}
            metadatas.append(metadata)

        if not embeddings:
            logger.warning("No valid embeddings found in chunks")
            return

        try:
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(embeddings)} chunks to vector database")
        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {e}")
            raise

    def search(
        self, 
        query_embedding: List[float], 
        n_results: int = 5,
        source_filter: Optional[str] = None,
        source_boost: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks in the vector database.

        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            source_filter: Filter results by source ('openbis' or 'datastore')
            source_boost: Boost results from this source

        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # Prepare where clause for filtering
            where_clause = None
            if source_filter:
                where_clause = {"source": source_filter}

            # If we want to boost a specific source, we'll do two searches
            if source_boost and not source_filter:
                # First search: get results from the boosted source
                boosted_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(1, n_results // 2),  # Get half from boosted source
                    where={"source": source_boost}
                )
                
                # Second search: get remaining results from all sources
                remaining_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=None
                )
                
                # Combine and deduplicate results
                all_ids = set()
                combined_results = {
                    'ids': [[]],
                    'distances': [[]],
                    'documents': [[]],
                    'metadatas': [[]]
                }
                
                # Add boosted results first
                for i, doc_id in enumerate(boosted_results['ids'][0]):
                    if doc_id not in all_ids:
                        all_ids.add(doc_id)
                        combined_results['ids'][0].append(doc_id)
                        combined_results['distances'][0].append(boosted_results['distances'][0][i])
                        combined_results['documents'][0].append(boosted_results['documents'][0][i])
                        combined_results['metadatas'][0].append(boosted_results['metadatas'][0][i])
                
                # Add remaining results
                for i, doc_id in enumerate(remaining_results['ids'][0]):
                    if doc_id not in all_ids and len(combined_results['ids'][0]) < n_results:
                        all_ids.add(doc_id)
                        combined_results['ids'][0].append(doc_id)
                        combined_results['distances'][0].append(remaining_results['distances'][0][i])
                        combined_results['documents'][0].append(remaining_results['documents'][0][i])
                        combined_results['metadatas'][0].append(remaining_results['metadatas'][0][i])
                
                results = combined_results
            else:
                # Standard search
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause
                )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                chunk = {
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    **results['metadatas'][0][i]  # Include all metadata
                }
                formatted_results.append(chunk)

            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get a sample of documents to analyze sources
            sample_results = self.collection.get(limit=min(100, count))
            sources = {}
            
            for metadata in sample_results.get('metadatas', []):
                source = metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'source_distribution': sources
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def reset_collection(self) -> None:
        """
        Reset the collection (delete all data).
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DeSi documentation chunks with embeddings"}
            )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def close(self) -> None:
        """
        Close the database connection.
        """
        # ChromaDB doesn't require explicit closing, but we can log it
        logger.info("Vector database connection closed")
