#!/usr/bin/env python3
"""
Configuration management for DeSi.

This module provides centralized configuration management with support for
environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class DesiConfig:
    """Centralized configuration for DeSi."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file (optional)
        """
        # Load .env file if available
        if DOTENV_AVAILABLE:
            if env_file:
                load_dotenv(env_file)
            else:
                # Try to load .env from project root
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                env_path = project_root / ".env"
                if env_path.exists():
                    load_dotenv(env_path)

    # Database Configuration
    @property
    def db_path(self) -> str:
        """Path to ChromaDB database directory."""
        return os.getenv("DESI_DB_PATH", "desi_vectordb")

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name."""
        return os.getenv("DESI_COLLECTION_NAME", "desi_docs")

    @property
    def memory_db_path(self) -> str:
        """Path to SQLite conversation memory database."""
        return os.getenv("DESI_MEMORY_DB_PATH", "desi_conversation_memory.db")

    # Model Configuration
    @property
    def model_name(self) -> str:
        """Ollama model name for chat."""
        return os.getenv("DESI_MODEL_NAME", "gpt-oss:20b")

    @property
    def embedding_model_name(self) -> str:
        """Ollama model name for embeddings."""
        return os.getenv("DESI_EMBEDDING_MODEL_NAME", "nomic-embed-text")

    # Data Configuration
    @property
    def data_dir(self) -> str:
        """Data directory path."""
        return os.getenv("DESI_DATA_DIR", "data")

    @property
    def processed_data_dir(self) -> str:
        """Processed data directory path."""
        return os.getenv("DESI_PROCESSED_DATA_DIR", "data/processed")

    # Scraper Configuration
    @property
    def openbis_url(self) -> str:
        """OpenBIS ReadTheDocs URL."""
        return os.getenv(
            "DESI_OPENBIS_URL",
            "https://openbis.readthedocs.io/en/20.10.0-11/index.html",
        )

    @property
    def wikijs_url(self) -> str:
        """Wiki.js DataStore URL."""
        return os.getenv("DESI_WIKIJS_URL", "https://datastore.bam.de/en/home")

    @property
    def max_pages_per_scraper(self) -> Optional[int]:
        """Maximum pages per scraper (None for unlimited)."""
        value = os.getenv("DESI_MAX_PAGES_PER_SCRAPER")
        return int(value) if value and value.isdigit() else None

    # Processing Configuration
    @property
    def min_chunk_size(self) -> int:
        """Minimum chunk size in characters."""
        return int(os.getenv("DESI_MIN_CHUNK_SIZE", "100"))

    @property
    def max_chunk_size(self) -> int:
        """Maximum chunk size in characters."""
        return int(os.getenv("DESI_MAX_CHUNK_SIZE", "1000"))

    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap in characters."""
        return int(os.getenv("DESI_CHUNK_OVERLAP", "50"))

    # Query Configuration
    @property
    def retrieval_top_k(self) -> int:
        """Number of chunks to retrieve for RAG."""
        return int(os.getenv("DESI_RETRIEVAL_TOP_K", "5"))

    @property
    def history_limit(self) -> int:
        """Number of conversation turns to keep in memory."""
        return int(os.getenv("DESI_HISTORY_LIMIT", "20"))

    # Web Configuration
    @property
    def web_host(self) -> str:
        """Web interface host."""
        return os.getenv("DESI_WEB_HOST", "127.0.0.1")

    @property
    def web_port(self) -> int:
        """Web interface port."""
        return int(os.getenv("DESI_WEB_PORT", "5000"))

    @property
    def web_debug(self) -> bool:
        """Enable web debug mode."""
        return os.getenv("DESI_WEB_DEBUG", "false").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

    @property
    def secret_key(self) -> str:
        """Flask secret key."""
        return os.getenv("DESI_SECRET_KEY", "desi_secret_key_change_in_production")

    @property
    def cors_origins(self) -> str:
        """CORS allowed origins."""
        return os.getenv("DESI_CORS_ORIGINS", "*")

    # Logging Configuration
    @property
    def log_level(self) -> str:
        """Logging level."""
        return os.getenv("DESI_LOG_LEVEL", "INFO")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "db_path": self.db_path,
            "collection_name": self.collection_name,
            "memory_db_path": self.memory_db_path,
            "model_name": self.model_name,
            "embedding_model_name": self.embedding_model_name,
            "data_dir": self.data_dir,
            "processed_data_dir": self.processed_data_dir,
            "openbis_url": self.openbis_url,
            "wikijs_url": self.wikijs_url,
            "max_pages_per_scraper": self.max_pages_per_scraper,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "history_limit": self.history_limit,
            "web_host": self.web_host,
            "web_port": self.web_port,
            "web_debug": self.web_debug,
            "secret_key": self.secret_key,
            "cors_origins": self.cors_origins,
            "log_level": self.log_level,
        }
