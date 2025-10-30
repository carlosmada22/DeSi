#!/usr/bin/env python3
"""
Main Conversation Engine for the Chatbot using LangGraph.

This script orchestrates the conversational workflow, managing user interaction,
short-term memory with SQLite, and interfacing with the RAG query engine
within a structured, extensible LangGraph graph.
"""

import logging
import sqlite3
import uuid
from typing import Dict, List, Tuple, TypedDict

from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

# Import the RAG engine from your existing query.py file
from .query import OLLAMA_AVAILABLE, RAGQueryEngine

# --- Basic Configuration ---
logger = logging.getLogger(__name__)


class SqliteConversationMemory:
    """
    SQLite-based conversation memory for storing and managing short-term context.
    """

    def __init__(
        self,
        db_path: str = "./data/chatbot_memory.db",
        history_limit: int = 20,
    ):
        self.db_path = db_path
        self.history_limit = history_limit
        self._init_database()
        logger.info(f"Connected to SQLite memory database at {db_path}")

    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON conversations(session_id, timestamp)
                """
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.commit()

    def get_recent_messages(self, session_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM conversations WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, self.history_limit),
            )
            rows = cursor.fetchall()
            return [
                {"role": role, "content": content} for role, content in reversed(rows)
            ]

    def clear_session(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            conn.commit()
        logger.info(f"Cleared memory for session: {session_id}")


class GraphState(TypedDict):
    """
    Represents the state of our conversation graph.

    Attributes:
        session_id: The unique ID for the conversation.
        user_query: The latest query from the user.
        history: The recent conversation history.
        response: The final response from the assistant.
        sources: A list of sources used by the RAG engine.
    """

    session_id: str
    user_query: str
    rewritten_query: str
    history: List[Dict]
    response: str
    sources: List[Dict]


class ChatbotEngine:
    """
    The main engine to run the chatbot conversation using a LangGraph workflow.
    """

    def __init__(
        self,
        rag_engine: RAGQueryEngine,
        memory: SqliteConversationMemory,
        rewrite_llm: ChatOllama,
    ):
        self.rag_engine = rag_engine
        self.memory = memory
        self.rewrite_llm = rewrite_llm
        self.graph: Pregel = self._build_graph()
        logger.info("ChatbotEngine with LangGraph workflow initialized.")

    def _format_history_for_prompt(self, history: List[Dict]) -> str:
        if not history:
            return ""
        formatted_history = "Previous conversation history:\n"
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {message['content']}\n"
        return formatted_history + "\n---\n"

    def rewrite_query_node(self, state: GraphState) -> Dict:
        """
        Node that rewrites the user's query to be self-contained for better retrieval.
        """
        logger.info(f"Node: rewrite_query_node for session {state['session_id']}")
        user_query = state["user_query"]
        history = self.memory.get_recent_messages(state["session_id"])

        if not history:  # If it's the first message, no need to rewrite
            return {"rewritten_query": user_query}

        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        rewrite_prompt = f"""
Given the following conversation history and a user's follow-up question, rewrite the user's question to be a standalone question that can be understood without the context of the chat history.

This standalone question will be used to search a knowledge base.

If the user's question is already a good standalone question, simply return it as is.

<Conversation History>
{history_str}
</Conversation History>

User's Follow-up Question: "{user_query}"

Standalone Question:
"""

        rewritten_query = self.rewrite_llm.invoke(rewrite_prompt).content
        logger.info(
            f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'"
        )
        return {"rewritten_query": rewritten_query}

    # --- Node Definitions for the Graph ---

    def call_rag_engine(self, state: GraphState) -> Dict:
        """
        Node that queries the RAG engine, providing history and query separately.
        """
        logger.info(f"Node: call_rag_engine for session {state['session_id']}")
        rewritten_query = state["rewritten_query"]
        session_id = state["session_id"]

        # 1. Get history from memory to provide conversational context
        history = self.memory.get_recent_messages(session_id)

        # 2. Call the RAG engine with separate arguments
        #    - `user_query` is used for clean vector retrieval.
        #    - `history` is used for LLM prompt context.
        answer, sources_docs = self.rag_engine.query(
            query=rewritten_query, conversation_history=history
        )
        sources = [doc.metadata for doc in sources_docs]

        return {"response": answer, "sources": sources}

    def update_memory(self, state: GraphState) -> Dict:
        """
        Node that saves the latest user query and assistant response to memory.
        """
        logger.info(f"Node: update_memory for session {state['session_id']}")
        self.memory.add_message(state["session_id"], "user", state["user_query"])
        self.memory.add_message(state["session_id"], "assistant", state["response"])

        return {}  # No state update needed from this node

    def _build_graph(self) -> Pregel:
        """
        Builds and compiles the LangGraph conversation workflow.
        """
        workflow = StateGraph(GraphState)

        # Add the nodes to the graph
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("rag_agent", self.call_rag_engine)
        workflow.add_node("update_memory", self.update_memory)

        # Define the flow of the graph
        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "rag_agent")
        workflow.add_edge("rag_agent", "update_memory")
        workflow.add_edge("update_memory", END)

        # Compile the graph into a runnable object
        return workflow.compile()

    def chat(self, user_input: str, session_id: str) -> Tuple[str, List[Dict]]:
        """
        Processes a single user message through the LangGraph workflow.
        """
        initial_state = GraphState(
            session_id=session_id,
            user_query=user_input,
            rewritten_query="",
            history=[],
            response="",
            sources=[],
        )
        # Invoke the graph with the initial state
        final_state = self.graph.invoke(initial_state)
        response = final_state.get("response", "Sorry, something went wrong.")
        sources = final_state.get("sources", [])
        return response, sources

    def _generate_initial_greeting(self) -> str:
        logger.info("Generating initial greeting from LLM...")
        greeting_prompt = (
            "You are **DeSi**, a friendly and expert assistant specializing **exclusively** in the BAM Data Store Project (mainly) and openBIS."
            "Generate a brief, welcoming opening message. Greet the user, introduce yourself and invite the user to ask a question."
        )
        try:
            # We create a dummy prompt without history or context for the greeting
            prompt = self.rag_engine._create_prompt(greeting_prompt, [], [])
            greeting = self.rag_engine.generate_answer(prompt)
            return greeting
        except Exception as e:
            logger.error(f"Failed to generate LLM greeting: {e}")
            return "Hello! I am DeSi. How can I help you today?"

    def start_chat_session(self):
        """
        Starts an interactive chat session in the terminal.
        """
        session_id = str(uuid.uuid4())
        print("--- Chat Session Started (LangGraph Workflow) ---")
        print(f"Session ID: {session_id}")
        print("Type 'exit' to end the session.\n")

        initial_greeting = self._generate_initial_greeting()
        print(f"Assistant: {initial_greeting}\n")
        self.memory.add_message(session_id, "assistant", initial_greeting)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower().strip() == "exit":
                    self.memory.clear_session(session_id)
                    print("Session ended and memory cleared. Goodbye!")
                    break
                if not user_input.strip():
                    continue

                # 1. Get both response and sources from the chat method
                response, sources = self.chat(user_input, session_id)

                # 2. Print the results using the desired formatting logic
                print(f"\nAssistant: {response}")
                print("\n--- Sources Used ---")
                displayed_sources = set()
                if not sources:
                    print("No sources were used for this response.")
                else:
                    for source_meta in sources:
                        source = source_meta.get("source", "N/A")
                        if source not in displayed_sources:
                            raw_origin = source_meta.get("origin", "N/A")
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
                self.memory.clear_session(session_id)
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                print("Sorry, an error occurred. Please check the logs.")


if __name__ == "__main__":
    # --- Configuration ---
    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"
    PROMPT_TEMPLATE_PATH = "./prompts/desi_query_prompt.md"
    SQLITE_DB_PATH = "./data/conversation_memory.db"
    CONVERSATION_HISTORY_LIMIT = 20
    # Value for boosting dswiki chunks
    DSWIKI_BOOST_VALUE = 0.2
    # A score of 0.7 means we discard any chunk with less than 70% similarity.
    RELEVANCE_THRESHOLD = 0.7
    # The model used by your RAG engine
    LLM_MODEL = "qwen3"

    if not OLLAMA_AVAILABLE:
        print(
            "Ollama is not available. Please start the Ollama server to run the chatbot."
        )
    else:
        try:
            rag_engine = RAGQueryEngine(
                chroma_persist_directory=CHROMA_PERSIST_DIRECTORY,
                dswiki_boost=DSWIKI_BOOST_VALUE,
                llm_model=LLM_MODEL,
                prompt_template_path=PROMPT_TEMPLATE_PATH,
                relevance_score_threshold=RELEVANCE_THRESHOLD,
            )
            conversation_memory = SqliteConversationMemory(
                db_path=SQLITE_DB_PATH, history_limit=CONVERSATION_HISTORY_LIMIT
            )
            # A separate, base LLM instance for the rewriting task
            rewrite_llm = ChatOllama(model=LLM_MODEL)

            # Initialize the chatbot engine
            chatbot = ChatbotEngine(
                rag_engine=rag_engine,
                memory=conversation_memory,
                rewrite_llm=rewrite_llm,
            )
            chatbot.start_chat_session()
        except Exception as e:
            logger.error(f"Failed to initialize the chatbot: {e}", exc_info=True)
