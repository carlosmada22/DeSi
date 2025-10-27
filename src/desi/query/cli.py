"""
Command-line interface for the DeSi Conversational Chatbot Engine.
"""

import argparse
import logging
import sys

from langchain_community.chat_models import ChatOllama

# Import the new, class-based components
from .conversation_engine import ChatbotEngine, SqliteConversationMemory
from .query import OLLAMA_AVAILABLE, RAGQueryEngine

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Run the DeSi RAG chatbot in your terminal."
    )
    # --- Arguments mapped to the new class constructors ---
    parser.add_argument(
        "--db-path",
        default="./desi_vectordb",
        help="Path to the ChromaDB vector store directory.",
    )
    parser.add_argument(
        "--prompt-template",
        default="./prompts/desi_query_prompt.md",
        help="Path to the prompt template file.",
    )
    parser.add_argument(
        "--memory-db-path",
        default="./data/conversation_memory.db",
        help="Path to the SQLite database for conversation memory.",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3",
        help="The Ollama model to use for generation and rewriting.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.7,
        help="The minimum similarity score for a chunk to be considered relevant.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not OLLAMA_AVAILABLE:
        logger.error(
            "Ollama is not available. Please start the Ollama server to run the chatbot."
        )
        sys.exit(1)

    try:
        # --- Instantiate all the components ---

        # 1. The core RAG engine for retrieval and generation
        rag_engine = RAGQueryEngine(
            chroma_persist_directory=args.db_path,
            prompt_template_path=args.prompt_template,
            llm_model=args.llm_model,
            relevance_score_threshold=args.relevance_threshold,
        )

        # 2. The memory system for conversation history
        conversation_memory = SqliteConversationMemory(db_path=args.memory_db_path)

        # 3. A separate LLM instance for the query rewriting task
        rewrite_llm = ChatOllama(model=args.llm_model)

        # 4. The main chatbot engine that orchestrates everything
        chatbot = ChatbotEngine(
            rag_engine=rag_engine,
            memory=conversation_memory,
            rewrite_llm=rewrite_llm,
        )

        # --- Start the interactive session ---
        chatbot.start_chat_session()

    except Exception as e:
        logger.error(f"Failed to initialize and run the chatbot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
