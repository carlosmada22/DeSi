#!/usr/bin/env python3
"""
Web interface for DeSi using FastAPI and Gradio.

This module provides a modern web interface for interacting with DeSi
through a browser-based chat interface powered by Gradio, with a FastAPI
backend for robust API handling.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel

from ..query.conversation_engine import ChatbotEngine, SqliteConversationMemory
from ..query.query import RAGQueryEngine
from ..utils.config import DesiConfig

# Configure logging
logger = logging.getLogger(__name__)

# Global conversation engine - initialized once on startup
conversation_engine: Optional[ChatbotEngine] = None
config: Optional[DesiConfig] = None
api_base_url: str = "http://127.0.0.1:7860"  # Default, will be updated on startup


# Pydantic models for API
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    session_id: str
    sources: List[dict]


class ClearSessionRequest(BaseModel):
    """Request model for clearing session."""

    session_id: str


def init_conversation_engine() -> ChatbotEngine:
    """
    Initialize the conversation engine on application startup.
    This is called only once when the FastAPI server starts.
    """
    global config
    config = DesiConfig()

    logger.info("🚀 Initializing DeSi Conversation Engine...")

    try:
        # Get project root directory
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        db_path = str(project_root / config.db_path)
        memory_db_path = str(project_root / config.memory_db_path)
        prompt_template_path = str(project_root / "prompts" / "desi_query_prompt.md")

        logger.info(f"Database path: {db_path}")
        logger.info(f"Memory database path: {memory_db_path}")
        logger.info(f"Prompt template path: {prompt_template_path}")

        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGQueryEngine(
            chroma_persist_directory=db_path,
            prompt_template_path=prompt_template_path,
            embedding_model=config.embedding_model_name,
            llm_model=config.model_name,
        )

        # Initialize conversation memory
        logger.info("Initializing conversation memory...")
        memory = SqliteConversationMemory(
            db_path=memory_db_path,
            history_limit=config.history_limit,
        )

        # Initialize rewrite LLM
        logger.info("Initializing rewrite LLM...")
        rewrite_llm = ChatOllama(model=config.model_name)

        # Create the chatbot engine
        logger.info("Creating chatbot engine...")
        engine = ChatbotEngine(
            rag_engine=rag_engine,
            memory=memory,
            rewrite_llm=rewrite_llm,
        )

        logger.info("✅ Chatbot Engine initialized successfully!")
        return engine

    except Exception as e:
        logger.error(f"❌ Failed to initialize conversation engine: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize conversation engine: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes the conversation engine on startup.
    """
    global conversation_engine

    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        conversation_engine = init_conversation_engine()
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down FastAPI application...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="DeSi - DataStore Helper",
    description="RAG-focused chatbot for openBIS and DataStore documentation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wwwtest.datastore.bam.de/en/"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if conversation_engine is None:
        raise HTTPException(
            status_code=503, detail="Conversation engine not initialized"
        )
    return {"status": "healthy", "engine": "ready"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for processing user messages.

    Args:
        request: ChatRequest containing the user message and optional session_id

    Returns:
        ChatResponse with the bot's response, session_id, and sources
    """
    if conversation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation engine not initialized",
        )

    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(
            f"Processing message for session {session_id}: {user_message[:50]}..."
        )

        # Process the message through the conversation engine
        response, sources = conversation_engine.chat(user_message, session_id)

        # Format sources for response
        formatted_sources = [
            {
                "source": doc.get("source", "N/A"),
                "origin": doc.get("origin", "N/A"),
            }
            for doc in sources
        ]

        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=formatted_sources,
        )

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}",
        )


@app.post("/api/clear-session")
async def clear_session_endpoint(request: ClearSessionRequest):
    """
    Clear conversation memory for a specific session.

    Args:
        request: ClearSessionRequest containing the session_id

    Returns:
        Success message
    """
    if conversation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation engine not initialized",
        )

    try:
        conversation_engine.memory.clear_session(request.session_id)
        logger.info(f"Cleared session: {request.session_id}")
        return {
            "message": "Session cleared successfully",
            "session_id": request.session_id,
        }

    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}",
        )


# ============================================================================
# Gradio Interface
# ============================================================================


def format_sources_display(sources: List[dict]) -> str:
    """
    Format sources for display in the Gradio interface.

    Args:
        sources: List of source dictionaries

    Returns:
        Formatted string for display
    """
    if not sources:
        return "\n\n📚 **Sources:** No sources were used for this response."

    # Remove duplicates based on source URL
    unique_sources = {}
    for source in sources:
        source_url = source.get("source", "N/A")
        if source_url not in unique_sources:
            unique_sources[source_url] = source

    # Format for display
    sources_text = "\n\n📚 **Sources:**\n"
    for source in unique_sources.values():
        origin = source.get("origin", "N/A")
        source_url = source.get("source", "N/A")

        # Make origin name more friendly
        if origin == "dswiki":
            display_origin = "DataStore Wiki"
        elif origin == "openbis":
            display_origin = "openBIS Wiki"
        else:
            display_origin = origin.title()

        sources_text += f"- **{display_origin}**: {source_url}\n"

    return sources_text


def chat_with_desi(
    message: str,
    history: List[Tuple[str, str]],
    session_id: str,
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Gradio chat function that communicates with the FastAPI backend.

    Args:
        message: User's input message
        history: Current chat history (list of [user_msg, bot_msg] pairs)
        session_id: Current session ID

    Returns:
        Tuple of (updated_history, session_id)
    """
    if not message or not message.strip():
        return history, session_id

    try:
        # Make request to FastAPI backend
        response = requests.post(
            f"{api_base_url}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=120,  # 2 minute timeout for LLM processing
        )
        response.raise_for_status()

        data = response.json()
        bot_response = data["response"]
        sources = data.get("sources", [])
        new_session_id = data["session_id"]

        # Format the bot response with sources
        full_response = bot_response + format_sources_display(sources)

        # Update history
        history.append((message, full_response))

        return history, new_session_id

    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out. The model might be taking too long to respond. Please try again."
        history.append((message, error_msg))
        return history, session_id

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Error communicating with backend: {str(e)}"
        logger.error(f"Request error: {e}", exc_info=True)
        history.append((message, error_msg))
        return history, session_id

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        history.append((message, error_msg))
        return history, session_id


def clear_conversation(session_id: str) -> Tuple[List, str, str]:
    """
    Clear the conversation history and start a new session.

    Args:
        session_id: Current session ID

    Returns:
        Tuple of (empty_history, new_session_id, welcome_message)
    """
    try:
        # Clear the session in the backend
        if session_id:
            requests.post(
                f"{api_base_url}/api/clear-session",
                json={"session_id": session_id},
                timeout=10,
            )

        # Generate new session ID
        new_session_id = str(uuid.uuid4())
        logger.info(f"Started new session: {new_session_id}")

        # Return empty history and welcome message
        welcome_msg = (
            "🔄 Conversation cleared! Starting a new session. How can I help you?"
        )
        return [], new_session_id, welcome_msg

    except Exception as e:
        logger.error(f"Error clearing conversation: {e}", exc_info=True)
        # Still create new session even if backend clear fails
        new_session_id = str(uuid.uuid4())
        return (
            [],
            new_session_id,
            "⚠️ Session cleared (with warnings). How can I help you?",
        )


def create_gradio_interface() -> gr.Blocks:
    """
    Create and configure the Gradio chat interface.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(
        title="DeSi - DataStore Helper",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """,
    ) as demo:
        # Session state
        session_id_state = gr.State(value=str(uuid.uuid4()))

        # Header
        gr.Markdown(
            """
            # 🤖 DeSi - DataStore Helper

            Your expert assistant for **openBIS** and **BAM DataStore** documentation.

            Ask me anything about these systems, and I'll provide detailed answers based on the official documentation!
            """
        )

        WELCOME_MESSAGE = "Hello! I'm DeSi, your assistant for openBIS and DataStore. How can I help you today?"

        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat with DeSi",
            height=500,
            show_label=True,
            avatar_images=(None, "🤖"),
            value=[(None, WELCOME_MESSAGE)],
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="Your message",
                placeholder="Type your question here... (e.g., 'How do I create a new space in openBIS?')",
                lines=2,
                scale=4,
                elem_id="chat-input",
            )
            send_btn = gr.Button(
                "Send 📤",
                variant="primary",
                scale=1,
                elem_id="chat-send-btn",
            )

        with gr.Row():
            clear_btn = gr.Button("🔄 Clear Conversation", variant="secondary")

        # Status message
        status_msg = gr.Textbox(
            label="Status",
            value="Ready! Ask me anything about openBIS or DataStore.",
            interactive=False,
            show_label=False,
        )

        # Session info (hidden but useful for debugging)
        with gr.Accordion("Session Info", open=False):
            gr.Textbox(
                label="Session ID",
                value=lambda: session_id_state.value,
                interactive=False,
            )

        # Event handlers
        def submit_message(message, history, session_id):
            """Handle message submission."""
            new_history, new_session_id = chat_with_desi(message, history, session_id)
            return "", new_history, new_session_id

        # Send button click
        send_btn.click(
            fn=submit_message,
            inputs=[msg_input, chatbot, session_id_state],
            outputs=[msg_input, chatbot, session_id_state],
        )

        # Enter key press
        msg_input.submit(
            fn=submit_message,
            inputs=[msg_input, chatbot, session_id_state],
            outputs=[msg_input, chatbot, session_id_state],
        )

        # Clear button
        clear_btn.click(
            fn=clear_conversation,
            inputs=[session_id_state],
            outputs=[chatbot, session_id_state, status_msg],
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This assistant uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on official documentation.
            Sources are displayed below each response.
            """
        )

    return demo


# Mount Gradio app to FastAPI
gradio_app = create_gradio_interface()
gr.mount_gradio_app(app, gradio_app, path="/")


# ============================================================================
# Main entry point for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration
    cfg = DesiConfig()

    # Set the API base URL for Gradio to communicate with FastAPI
    api_base_url = f"http://{cfg.web_host}:{cfg.web_port}"

    logger.info("🚀 Starting DeSi Web Interface...")
    logger.info(f"Host: {cfg.web_host}")
    logger.info(f"Port: {cfg.web_port}")
    logger.info(f"API Base URL: {api_base_url}")

    # Run with uvicorn
    uvicorn.run(
        "desi.web.app:app",
        host=cfg.web_host,
        port=cfg.web_port,
        reload=cfg.web_debug,
        log_level="info",
    )
