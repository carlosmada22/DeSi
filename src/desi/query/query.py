#!/usr/bin/env python3
"""
RAG Query Engine

This module provides the core functionality for querying the vector database
and generating answers using a Retrieval-Augmented Generation (RAG) pipeline.
It connects to a persistent ChromaDB vector store and uses Ollama for both
embedding and language model generation.
"""

import logging
from typing import List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# LangChain components are used for interacting with the vector store and the LLM
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
        llm_model: str = "gpt-oss",
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
        """
        self.chroma_persist_directory = chroma_persist_directory
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model

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
        Retrieves the most relevant document chunks for a given query from ChromaDB.

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

        logger.info(f"Performing similarity search for top {top_k} chunks...")
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Found {len(results)} relevant chunks.")
            return results
        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

    def _create_prompt(self, query: str, relevant_chunks: List[Document]) -> str:
        """
        Creates a detailed prompt for the LLM, including the query and context.

        Args:
            query (str): The user's query.
            relevant_chunks (List[Document]): The context chunks retrieved from the
                                               vector store.

        Returns:
            str: The fully formatted prompt for the LLM.
        """
        context_str = ""
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            context_str += f"--- Context Chunk {i} (Source: {source}) ---\n"
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

        Args:
            query (str): The user's query.
            relevant_chunks (List[Document]): The context to be used for the answer.

        Returns:
            str: The LLM-generated answer.
        """
        if not self.llm:
            return "The Language Model is not available. Cannot generate an answer."
        if not relevant_chunks:
            return "I do not have enough information to answer that question."

        prompt = self._create_prompt(query, relevant_chunks)
        logger.info("Generating answer with LLM...")

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"An error occurred while generating the answer: {e}")
            return "There was an error generating the answer."

    def query(self, query: str, top_k: int = 5) -> Tuple[str, List[Document]]:
        """
        Executes the full RAG pipeline for a given query.

        Args:
            query (str): The user's question.
            top_k (int): The number of documents to retrieve for context.

        Returns:
            Tuple[str, List[Document]]: A tuple containing the generated answer
                                         and the list of source documents used
                                         as context.
        """
        if not OLLAMA_AVAILABLE or not self.vector_store or not self.llm:
            error_message = "Cannot process query because a required component (Ollama, Vector Store, or LLM) is not available."
            logger.error(error_message)
            return error_message, []

        # Step 1: Retrieve relevant chunks from the vector database
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)

        # Step 2: Generate an answer using the retrieved context
        answer = self.generate_answer(query, relevant_chunks)

        return answer, relevant_chunks


if __name__ == "__main__":
    # --- Standalone Execution Example ---
    # This demonstrates how to use the RAGQueryEngine.
    # It assumes the ChromaDB is located in the specified directory.

    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"

    print("--- RAG Query Engine Initializing ---")
    query_engine = RAGQueryEngine(chroma_persist_directory=CHROMA_PERSIST_DIRECTORY)
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
                for doc in source_chunks:
                    source = doc.metadata.get("source", "N/A")
                    print(f"- Source: {source}")
                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        print(
            "Failed to initialize the RAG Query Engine. Please check the logs for errors."
        )
