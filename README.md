# DeSi: DataStore Helper

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/carlosmada22/DeSi)](https://github.com/carlosmada22/DeSi/issues)
[![GitHub stars](https://img.shields.io/github/stars/carlosmada22/DeSi)](https://github.com/carlosmada22/DeSi/stargazers)

DeSi: DataStore Helper, is a RAG-focused chatbot that provides intelligent assistance for openBIS and Data Store documentation. It sources information from two distinct knowledge bases: ReadTheDocs (openBIS documentation) and Wiki.js (Data Store wiki), using a vector database for efficient retrieval.

## Table of Contents

- [DeSi: DataStore Helper](#desi-datastore-helper)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
    - [Wiki.js URL](#wikijs-url)
    - [Other Settings](#other-settings)
  - [Usage](#usage)
    - [Command Line Interface](#command-line-interface)
    - [Web Interface](#web-interface)
  - [Architecture](#architecture)
  - [Example Queries](#example-queries)
  - [Directory Structure](#directory-structure)
  - [Development](#development)
    - [Adding New Scrapers](#adding-new-scrapers)
    - [Customizing Processing](#customizing-processing)
    - [Extending the Query Engine](#extending-the-query-engine)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Logging](#logging)
    - [Database Statistics](#database-statistics)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Multi-Source RAG**: Retrieves information from both openBIS ReadTheDocs and Wiki.js sources.
- **Vector Database**: Uses ChromaDB for efficient similarity search and retrieval.
- **Source Prioritization**: Prioritizes Data Store wiki content when relevant.
- **Conversation Memory**: Maintains context across multiple interactions.
- **Web & CLI Interfaces**: Interact via a browser-based chat or directly from the command line.
- **Modular Architecture**: Separate components for scraping, processing, and querying.

## Prerequisites

- Python 3.8+
- Ollama with `qwen3` and `nomic-embed-text` models installed.

To install the required Ollama models, run:
```bash
ollama pull qwen3
ollama pull nomic-embed-text
```

## Installation

It is highly recommended to use a virtual environment to manage project dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/carlosmada22/DeSi.git
    cd DeSi
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the project and its dependencies:**
    This single command installs DeSi in "editable" mode along with all the tools needed for development (like `pytest`, `ruff`, etc.). It reads all dependencies directly from the `pyproject.toml` file.
    ```bash
    pip install -e ".[dev]"
    ```

4.  **Prepare the environment:**
    After installation, run the initialization script to check for necessary services (like Ollama), download AI models, and create the required data directories.
    ```bash
    python init.py
    ```

## Quick Start

For the fastest start, run the complete, automated pipeline. This will scrape the data, build the database, and launch the query interface.

```bash
python main.py
```

You can now start asking questions like: *"How do I create a new experiment in openBIS?"*

## Configuration

### Wiki.js URL

Edit `src/desi/__main__.py` and update the `DEFAULT_WIKIJS_URL` variable:

```python
DEFAULT_WIKIJS_URL = "https://datastore.bam.de/en/home"
```

### Other Settings

You can customize various settings:

- **Database path**: `--db-path desi_vectordb`
- **Collection name**: `--collection-name desi_docs`
- **Ollama model**: `--model qwen3`
- **Chunk sizes**: `--min-chunk-size 100 --max-chunk-size 1000`

## Usage

### Command Line Interface

For direct interaction or scripting, use the CLI.

```bash
# Start an interactive query session
python main.py query

# Show database statistics
python main.py query --stats

# Use a different LLM model
python main.py query --model llama2
```

### Web Interface

For a user-friendly chat experience, run the web app.

```bash
# Start the web interface on the default port (5000)
python -m desi.web.cli

# Run on a custom host and port
python -m desi.web.cli --host 0.0.0.0 --port 8080
```
Then open your browser to `http://localhost:5000`.

## Architecture

DeSi consists of several key components:
1.  **Scrapers**: Extract content from ReadTheDocs and Wiki.js sites.
2.  **Processor**: Chunks content and generates embeddings using Ollama.
3.  **Vector Database**: ChromaDB for storing and retrieving document embeddings.
4.  **Query Engine**: A RAG pipeline with source prioritization logic.
5.  **Conversation Engine**: Manages chat history and maintains context.
6.  **Web Interface**: A Flask-based web application for user interaction.

## Example Queries

- "How do I create a new experiment in openBIS?"
- "How to log in to the BAM Data Store?"
- "What are the steps to register a collection?"
- "How do I upload data to the data store?"
- "What is the difference between spaces and projects in openBIS?"

## Directory Structure

```
DeSi/
├── src/desi/
│   ├── scraper/          # Web scrapers for different sources
│   ├── processor/        # Content processing and embedding generation
│   ├── query/           # RAG query engine and conversation management
│   ├── utils/           # Utilities (logging, vector database)
│   └── web/             # Web interface
├── scripts/             # Pipeline and utility scripts
├── tests/               # Unit tests and integration tests
├── data/               # Data storage (raw and processed)
├── pyproject.toml     # Project configuration and dependencies
├── AUTHORS            # Contributors list
├── LICENSE            # MIT License
├── init.py            # Environment setup and validation script
├── main.py            # Main entry point
└── README.md          # This file
```

## Development

### Adding New Scrapers

1. Create a new scraper class in `src/desi/scraper/`
2. Implement the required methods following existing patterns
3. Add CLI support in `src/desi/scraper/cli.py`
4. Update the main pipeline to include the new scraper

### Customizing Processing

Modify `src/desi/processor/processor.py` to:
- Change chunking strategies
- Add new metadata fields
- Implement custom embedding models

### Extending the Query Engine

Enhance `src/desi/query/query.py` to:
- Add new retrieval strategies
- Implement custom ranking algorithms
- Add support for different LLM models

## Troubleshooting

### Common Issues

1. **Ollama not available**: Ensure Ollama is running and models are installed
2. **Empty vector database**: Run the ingestion step with `--reset` flag
3. **Web interface not loading**: Check that Flask dependencies are installed
4. **Scraping fails**: Verify URLs are accessible and sites are online

### Logging

Enable verbose logging for debugging:

```bash
python main.py query --verbose
```

### Database Statistics

Check database contents:

```bash
python main.py query --stats
```

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or suggest a feature. If you would like to contribute code, please open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
