# Configuration

DeSi reads settings from environment variables, optionally loaded from a `.env` file by `DesiConfig`. CLI flags that accept `--config` pass a custom env file path; environment variables always win over file values.

## Use a .env File
1. Copy the template: `.env.example` → `.env` (or run `python init.py`).
2. Edit values, for example:
   ```
   DESI_DB_PATH=desi_vectordb
   DESI_MODEL_NAME=gpt-oss:20b
   DESI_EMBEDDING_MODEL_NAME=nomic-embed-text
   DESI_WEB_PORT=5000
   ```
3. Run commands normally; `DesiConfig` loads `.env` from the project root automatically.

## Override with Environment Variables
Any variable can be set per command:
```bash
set DESI_MODEL_NAME=qwen3
python main.py
```
Explicit environment variables override `.env` values.

## Point to a Custom Config File
Several CLIs accept `--config <file>`:
```bash
python main.py --config ./staging.env
python -m desi.web.cli --config ./staging.env --reload
```
`DesiConfig` loads that file first, then applies environment overrides.

## What DesiConfig Provides
See [Configuration Reference](../reference/configuration.md) for the full variable list, defaults, and meanings drawn directly from `src/desi/utils/config.py`.
