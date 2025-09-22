#!/usr/bin/env python3
"""
Web interface for DeSi using Flask.

This module provides a simple web interface for interacting with DeSi
through a browser-based chat interface.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

from ..query.conversation_engine import DesiConversationEngine

# Configure logging
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = "desi_secret_key_change_in_production"  # Change this in production
CORS(app)

# Global conversation engine
conversation_engine = None


def init_conversation_engine(db_path: str, collection_name: str = "desi_docs", model: str = "gpt-oss:20b"):
    """Initialize the conversation engine."""
    global conversation_engine
    try:
        conversation_engine = DesiConversationEngine(
            db_path=db_path,
            collection_name=collection_name,
            model=model
        )
        logger.info("Conversation engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize conversation engine: {e}")
        return False


@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    if not conversation_engine:
        return jsonify({
            'error': 'Conversation engine not initialized',
            'response': 'Sorry, the system is not ready. Please check the server logs.'
        }), 500

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get or create session ID
        session_id = session.get('session_id')
        
        # Process the message
        response, session_id, metadata = conversation_engine.chat(user_message, session_id)
        
        # Store session ID
        session['session_id'] = session_id
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return jsonify({
            'error': 'Internal server error',
            'response': f'Sorry, I encountered an error: {str(e)}'
        }), 500


@app.route('/api/history')
def get_history():
    """Get conversation history for the current session."""
    if not conversation_engine:
        return jsonify({'error': 'Conversation engine not initialized'}), 500

    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'history': []})

        history = conversation_engine.get_conversation_history(session_id)
        return jsonify({'history': history})

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({'error': 'Failed to get conversation history'}), 500


@app.route('/api/stats')
def get_stats():
    """Get database statistics."""
    if not conversation_engine:
        return jsonify({'error': 'Conversation engine not initialized'}), 500

    try:
        stats = conversation_engine.get_database_stats()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({'error': 'Failed to get database stats'}), 500


@app.route('/api/new-session', methods=['POST'])
def new_session():
    """Start a new conversation session."""
    try:
        # Clear the current session
        session.pop('session_id', None)

        return jsonify({'message': 'New session started'})

    except Exception as e:
        logger.error(f"Error starting new session: {e}")
        return jsonify({'error': 'Failed to start new session'}), 500


@app.route('/api/session-stats')
def get_session_stats():
    """Get statistics for the current session."""
    if not conversation_engine:
        return jsonify({'error': 'Conversation engine not initialized'}), 500

    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'stats': {'session_id': None, 'total_messages': 0, 'total_tokens': 0}})

        stats = conversation_engine.get_session_stats(session_id)
        return jsonify({'stats': stats})

    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        return jsonify({'error': 'Failed to get session stats'}), 500


@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear the current session's conversation memory."""
    if not conversation_engine:
        return jsonify({'error': 'Conversation engine not initialized'}), 500

    try:
        session_id = session.get('session_id')
        if session_id:
            success = conversation_engine.clear_session_memory(session_id)
            if success:
                return jsonify({'message': 'Session memory cleared successfully'})
            else:
                return jsonify({'error': 'Failed to clear session memory'}), 500
        else:
            return jsonify({'message': 'No active session to clear'})

    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': 'Failed to clear session'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def create_app(db_path: str, collection_name: str = "desi_docs", model: str = "gpt-oss:20b"):
    """Create and configure the Flask app."""
    # Initialize conversation engine
    if not init_conversation_engine(db_path, collection_name, model):
        raise RuntimeError("Failed to initialize conversation engine")
    
    return app


if __name__ == '__main__':
    # This is for development only
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DeSi web interface")
    parser.add_argument("--db-path", default="desi_vectordb", help="Path to ChromaDB database")
    parser.add_argument("--collection-name", default="desi_docs", help="ChromaDB collection name")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model to use")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize conversation engine
    if init_conversation_engine(args.db_path, args.collection_name, args.model):
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        print("Failed to initialize conversation engine. Exiting.")
        exit(1)
