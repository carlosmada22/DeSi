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
from typing import List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# We import the Document class for type hinting, as Chroma returns this type
from langchain_core.documents import Document

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "qwen3",
        dswiki_boost: float = 0.15,  # --- NEW: Tunable boost parameter ---
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
        """
        self.chroma_persist_directory = chroma_persist_directory
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.dswiki_boost = dswiki_boost  # --- NEW: Store boost value ---

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
                self.vector_store.similarity_search_with_score(
                    query, k=candidate_pool_size
                )
            )
        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

        # 2. Calculate an adjusted score for each document based on metadata.
        reranked_results = []
        for doc, score in initial_results_with_scores:
            adjusted_score = score
            if doc.metadata.get("origin") == "dswiki":
                adjusted_score -= self.dswiki_boost
                logger.debug(
                    f"Boosting dswiki doc '{doc.metadata.get('source')}'. Original: {score:.4f}, Boosted: {adjusted_score:.4f}"
                )

            reranked_results.append((doc, adjusted_score))

        # 3. Sort the entire pool based on the new, adjusted score in ascending order (lower is better).
        reranked_results.sort(key=lambda x: x[1], reverse=False)

        # 4. Extract just the documents from the sorted list and return the top_k.
        final_docs = [doc for doc, score in reranked_results[:top_k]]
        logger.info(
            f"Found {len(initial_results_with_scores)} candidates. Returning {len(final_docs)} re-ranked chunks."
        )

        return final_docs

    def _create_prompt(self, query: str, relevant_chunks: List[Document]) -> str:
        """
        Creates a detailed prompt for the LLM, including the query and context.
        """
        context_str = ""
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            origin = chunk.metadata.get("origin", "Unknown")
            context_str += (
                f"--- Context Chunk {i} (Origin: {origin}, Source: {source}) ---\n"
            )
            context_str += chunk.page_content
            context_str += "\n--------------------------------------------\n\n"

        prompt = f"""
You are an expert assistant for openBIS and DSWiki. Your goal is to provide clear, accurate, and friendly answers based ONLY on the context provided below.

Follow these rules STRICTLY:
1.  **Base your answer exclusively on the provided context.** Do not use any prior knowledge.
2.  **Do not mention the context or the documentation in your answer.** Never say "Based on the information provided..." or similar phrases.
3.  **Synthesize information** from all provided chunks to form a complete and coherent answer.
4.  If the context does not contain the answer, state that you do not have enough information to answer the question. Do not try to guess.
5.  Be conversational and helpful in your tone.

--- CONTEXT ---
{context_str}
--- END OF CONTEXT ---

Based on the context above, please provide a clear and helpful answer to the following question.

Question: {query}
Answer:
"""
        return prompt

    def generate_answer(self, query: str, relevant_chunks: List[Document]) -> str:
        """
        Generates an answer using the LLM based on the query and relevant chunks.
        """
        if not self.llm:
            return "The Language Model is not available. Cannot generate an answer."
        if not relevant_chunks:
            return "I do not have enough information to answer that question."

        prompt = self._create_prompt(query, relevant_chunks)
        logger.info("Generating answer with LLM...")

        try:
            response = self.llm.invoke(prompt)
            raw_answer = response.content

            # This regex finds the <think>...</think> block (including multi-line content)
            # and replaces it with an empty string. It will not error if the block is not found.
            cleaned_answer = re.sub(
                r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
            )

            return cleaned_answer.strip()

        except Exception as e:
            logger.error(f"An error occurred while generating the answer: {e}")
            return "There was an error generating the answer."

    def query(self, query: str, top_k: int = 5) -> Tuple[str, List[Document]]:
        """
        Executes the full RAG pipeline for a given query.
        """
        if not OLLAMA_AVAILABLE or not self.vector_store or not self.llm:
            error_message = "Cannot process query because a required component (Ollama, Vector Store, or LLM) is not available."
            logger.error(error_message)
            return error_message, []

        # Step 1: Retrieve relevant chunks from the vector database (with score boosting)
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)

        # Step 2: Generate an answer using the retrieved context
        answer = self.generate_answer(query, relevant_chunks)

        return answer, relevant_chunks


if __name__ == "__main__":
    # --- Standalone Execution Example ---
    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"
    # A value of 25.0 provides a significant but not absolute boost for L2 distance.
    # This value may need tuning depending on the embedding model.
    DSWIKI_BOOST_VALUE = 25.0

    print("--- RAG Query Engine Initializing ---")
    query_engine = RAGQueryEngine(
        chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
        dswiki_boost=DSWIKI_BOOST_VALUE,
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
