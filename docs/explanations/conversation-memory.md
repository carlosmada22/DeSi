# Conversation Memory

`SqliteConversationMemory` (`src/desi/query/conversation_engine.py`) stores chat history in `data/conversation_memory.db`.

- Schema: `conversations(id, session_id, role, content, timestamp)` with an index on `(session_id, timestamp)`.
- `history_limit` (default `20`) controls how many past turns are pulled into prompts.
- Messages are appended after every exchange; `clear_session(session_id)` deletes them.

The LangGraph workflow in `ChatbotEngine`:
1. Loads recent messages for the session.
2. Rewrites follow-up questions into standalone queries for better retrieval.
3. Saves both user and assistant messages after generating a response.

For the web API, sessions are keyed by a UUID returned in each `ChatResponse`. To reset context, call `POST /api/clear-session` or delete the SQLite file.
