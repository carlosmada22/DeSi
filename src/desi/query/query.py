#!/usr/bin/env python3
"""
RAG Query Engine

This module provides the core functionality for querying the vector database
and generating answers using a Retrieval-Augmented Generation (RAG) pipeline.
It connects to a persistent ChromaDB vector store and uses Ollama for both
embedding and language model generation.
"""

import logging
import re
from typing import Dict, List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma

# We import the Document class for type hinting, as Chroma returns this type
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# --- Basic Configuration ---
logger = logging.getLogger(__name__)


# --- OLLAMA AVAILABILITY CHECK ---
# A check to ensure the Ollama server is running before attempting to use it.
try:
    # Attempt to initialize the embedding model
    OllamaEmbeddings(model="nomic-embed-text")
    OLLAMA_AVAILABLE = True
    logger.info("Ollama server connection successful.")
except Exception as e:
    logger.warning(
        f"Could not connect to Ollama server. "
        f"Please ensure Ollama is running. Error: {e}"
    )
    OLLAMA_AVAILABLE = False


class RAGQueryEngine:
    """
    Manages the entire RAG pipeline from query to answer.
    """

    def __init__(
        self,
        chroma_persist_directory: str,
        prompt_template_path: str,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "qwen3",
        dswiki_boost: float = 0.15,
        relevance_score_threshold: float = 0.3,
    ):
        """
        Initializes the RAG query engine.

        Args:
            chroma_persist_directory (str): The directory where the ChromaDB
                                            vector store is persisted.
            embedding_model (str): The name of the Ollama model to use for
                                   generating embeddings.
            llm_model (str): The name of the Ollama model to use for generating
                             answers.
            dswiki_boost (float): A value to add to the relevance score of chunks
                                  from 'dswiki' to prioritize them.
            relevance_score_threshold (float): The minimum similarity score for a chunk
                                               to be considered relevant. Chunks with a
                                               score *below* this are discarded.
        """
        self.chroma_persist_directory = chroma_persist_directory
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.dswiki_boost = dswiki_boost
        self.relevance_score_threshold = relevance_score_threshold
        self.prompt_template = self._load_prompt_template(prompt_template_path)

        if not OLLAMA_AVAILABLE:
            logger.error("Cannot initialize RAGQueryEngine: Ollama is not available.")
            self.vector_store = None
            self.llm = None
            return

        logger.info("Initializing embedding model and vector store...")
        try:
            self.embedding_model = OllamaEmbeddings(model=self.embedding_model_name)
            self.vector_store = Chroma(
                persist_directory=self.chroma_persist_directory,
                embedding_function=self.embedding_model,
            )
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load vector store from '{chroma_persist_directory}'. Error: {e}"
            )
            self.vector_store = None

        logger.info("Initializing Large Language Model...")
        try:
            self.llm = ChatOllama(model=self.llm_model_name)
            logger.info(f"LLM '{self.llm_model_name}' initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM. Error: {e}")
            self.llm = None

    def _load_prompt_template(self, file_path: str) -> str:
        """Loads the prompt template from a file."""
        logger.info(f"Loading prompt template from '{file_path}'...")
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at '{file_path}'.")
            return ""  # Return empty or raise an exception
        except Exception as e:
            logger.error(f"Failed to load prompt template. Error: {e}")
            return ""

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves the most relevant document chunks for a given query from ChromaDB,
        applying a score boost to chunks from the 'dswiki' origin.

        Args:
            query (str): The user's query.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Document]: A list of LangChain Document objects containing the
                            retrieved content and metadata.
        """
        if not self.vector_store:
            logger.warning("Vector store is not available. Cannot retrieve chunks.")
            return []

        # 1. Fetch a larger candidate pool along with their relevance scores.
        #    Note: similarity_search_with_relevance_scores returns scores where
        #    *higher is better*.
        candidate_pool_size = top_k * 4
        logger.info(
            f"Fetching candidate pool of {candidate_pool_size} chunks with scores..."
        )
        try:
            # This method returns a list of (Document, score) tuples
            initial_results_with_scores = (
                self.vector_store.similarity_search_with_relevance_scores(
                    query, k=candidate_pool_size
                )
            )
        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

        # 2. We keep scores ABOVE the threshold.
        relevant_results = [
            (doc, score)
            for doc, score in initial_results_with_scores
            if score >= self.relevance_score_threshold
        ]
        discarded_count = len(initial_results_with_scores) - len(relevant_results)
        if discarded_count > 0:
            logger.info(
                f"Discarded {discarded_count} chunks below relevance threshold ({self.relevance_score_threshold})."
            )

        if not relevant_results:
            logger.info(
                "No chunks met the relevance threshold. Answering from persona."
            )
            return []

        # 3. Calculate an adjusted score for each document based on metadata.
        reranked_results = []
        for doc, score in relevant_results:
            adjusted_score = score
            if doc.metadata.get("origin") == "dswiki":
                adjusted_score += self.dswiki_boost
                logger.debug(
                    f"Boosting dswiki doc '{doc.metadata.get('source')}'. Original: {score:.4f}, Boosted: {adjusted_score:.4f}"
                )

            reranked_results.append((doc, adjusted_score))

        # 4. Sort the entire pool based on the new, adjusted score in ascending order (lower is better).
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        # 5. Extract just the documents from the sorted list and return the top_k.
        final_docs = [doc for doc, score in reranked_results[:top_k]]
        logger.info(
            f"Found {len(initial_results_with_scores)} candidates. Returning {len(final_docs)} re-ranked chunks."
        )

        return final_docs

    def _create_prompt(
        self,
        query: str,
        relevant_chunks: List[Document],
        conversation_history: List[Dict],
    ) -> str:
        """
        Creates a detailed prompt for the LLM, including the query and context.
        """
        if not self.prompt_template:
            logger.error("Prompt template is not loaded. Cannot create prompt.")
            return (
                "You are **DeSi**, a friendly and expert assistant specializing **exclusively** in the BAM Data Store Project (mainly) and openBIS (through the DSWiki and openBIS documentation). Your primary goal is to provide clear, accurate, and helpful answers to users' questions about these systems. You must be conversational, confident, and consistently knowledgeable."
                + query
            )

        context_str = ""
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks, 1):
                source = chunk.metadata.get("source", "Unknown")
                origin = chunk.metadata.get("origin", "Unknown")
                context_str += (
                    f"--- Context Chunk {i} (Origin: {origin}, Source: {source}) ---\n"
                )
                context_str += chunk.page_content
                context_str += "\n--------------------------------------------\n\n"
        else:
            context_str = "No specific documentation context was found for this query."

        # Format conversation history
        history_str = ""
        if conversation_history:
            for message in conversation_history:
                role = "User" if message["role"] == "user" else "Assistant"
                history_str += f"{role}: {message['content']}\n"
        else:
            history_str = "This is the beginning of the conversation."

        return self.prompt_template.format(
            history_str=history_str, context_str=context_str, query=query
        )

    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer using the LLM based on the query and relevant chunks.
        """
        if not self.llm:
            return "The Language Model is not available. Cannot generate an answer."

        logger.info("Generating answer with LLM...")

        try:
            response = self.llm.invoke(prompt)
            raw_answer = response.content
            cleaned_answer = re.sub(
                r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
            )
            return cleaned_answer.strip()
        except Exception as e:
            logger.error(f"An error occurred while generating the answer: {e}")
            return "There was an error generating the answer."

    def query(
        self, query: str, conversation_history: List[Dict] = None, top_k: int = 5
    ) -> Tuple[str, List[Document]]:
        """
        Executes the full RAG pipeline for a given query.
        """
        if not OLLAMA_AVAILABLE or not self.vector_store or not self.llm:
            error_message = "Cannot process query because a required component (Ollama, Vector Store, or LLM) is not available."
            logger.error(error_message)
            return error_message, []

        if conversation_history is None:
            conversation_history = []

        # Step 1: Retrieve relevant chunks from the vector database (with score boosting)
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)

        # Step 2: Create a rich prompt with history, context, and the query
        prompt = self._create_prompt(query, relevant_chunks, conversation_history)

        # Step 3: Generate an answer using the rich prompt
        answer = self.generate_answer(prompt)

        return answer, relevant_chunks


if __name__ == "__main__":
    # --- Standalone Execution Example ---
    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"
    # Value for boosting dswiki chunks
    DSWIKI_BOOST_VALUE = 0.15
    # A score of 0.3 means we discard any chunk with less than 30% similarity.
    RELEVANCE_THRESHOLD = 0.7
    # Path to the prompt template
    PROMPT_TEMPLATE_PATH = "./prompts/desi_query_prompt.md"

    print("--- RAG Query Engine Initializing ---")
    query_engine = RAGQueryEngine(
        chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
        dswiki_boost=DSWIKI_BOOST_VALUE,
        prompt_template_path=PROMPT_TEMPLATE_PATH,
        relevance_score_threshold=RELEVANCE_THRESHOLD,
    )
    print("-------------------------------------\n")

    if OLLAMA_AVAILABLE and query_engine.vector_store and query_engine.llm:
        print("Initialization successful. You can now ask questions.")
        print("Type 'exit' to quit the program.\n")

        while True:
            try:
                user_query = input("Ask a question: ")
                if user_query.lower().strip() == "exit":
                    print("Exiting...")
                    break
                if not user_query.strip():
                    continue

                # Execute the RAG pipeline
                final_answer, source_chunks = query_engine.query(user_query)

                # Print the results
                print("\n--- Answer ---\n")
                print(final_answer)
                print("\n--- Sources Used ---\n")
                displayed_sources = set()
                if not source_chunks:
                    print("No sources were used.")
                else:
                    for doc in source_chunks:
                        source = doc.metadata.get("source", "N/A")
                        if source not in displayed_sources:
                            # Make origin name more friendly for display
                            raw_origin = doc.metadata.get("origin", "N/A")
                            if raw_origin == "dswiki":
                                display_origin = "DataStore Wiki"
                            elif raw_origin == "openbis":
                                display_origin = "openBIS Wiki"
                            else:
                                display_origin = raw_origin.title()

                            print(f"- Origin: {display_origin}, Source: {source}")
                            displayed_sources.add(source)

                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        print(
            "Failed to initialize the RAG Query Engine. Please check the logs for errors."
        )
