# Run Web UI

Start the FastAPI + Gradio interface served by uvicorn.

## Command
```bash
python -m desi.web.cli [options]
```

### Arguments
- `--host` (default from `DesiConfig.web_host`, fallback `0.0.0.0`): bind address.
- `--port` (default from `DesiConfig.web_port`, fallback `5000`): listening port.
- `--reload`: enable auto-reload for development.
- `--config <file>`: optional `.env` path to load before reading environment variables.

## Example
```bash
python -m desi.web.cli --host 127.0.0.1 --port 7860 --reload
```
The CLI loads configuration, logs the resolved host/port, updates `desi.web.app.api_base_url`, and then runs `uvicorn desi.web.app:app`.

### Health Checks
- `GET /health`: verifies the conversation engine initialized.
- `POST /api/chat`: accepts `{ "message": "...", "session_id": "<optional>" }` and returns the model response plus sources.
- `POST /api/clear-session`: clears memory for a session.

If FastAPI or Gradio imports fail, the CLI reports the missing packages and exits.
