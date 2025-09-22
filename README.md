# DeSi

DeSi is a RAG-focused chatbot that provides intelligent assistance for openBIS and Data Store documentation. It sources information from two distinct knowledge bases: ReadTheDocs (openBIS documentation) and Wiki.js (Data Store wiki), using a vector database for efficient retrieval.

## Features

- **Multi-Source RAG**: Retrieves information from both openBIS ReadTheDocs and Wiki.js sources
- **Vector Database**: Uses ChromaDB for efficient similarity search and retrieval
- **Source Prioritization**: Prioritizes Data Store wiki content when relevant
- **Conversation Memory**: Maintains context across multiple interactions
- **Web Interface**: Browser-based chat interface for easy interaction
- **CLI Interface**: Command-line interface for direct queries
- **Modular Architecture**: Separate components for scraping, processing, and querying

## Architecture

DeSi consists of several key components:

1. **Scrapers**: Extract content from ReadTheDocs and Wiki.js sites
2. **Processor**: Chunks content and generates embeddings using Ollama
3. **Vector Database**: ChromaDB for storing and retrieving embeddings
4. **Query Engine**: RAG pipeline with source prioritization
5. **Conversation Engine**: Memory-enabled chat interface
6. **Web Interface**: Flask-based web application

## Prerequisites

- Python 3.8+
- Ollama with `qwen3` and `nomic-embed-text` models
- ChromaDB

### Installing Ollama Models

```bash
ollama pull qwen3
ollama pull nomic-embed-text
```

## Installation

1. Clone or copy the DeSi directory
2. Install dependencies:

```bash
cd DeSi
pip install -r requirements.txt
```

## Quick Start

### Option 1: Automatic Pipeline (Recommended)

Run the complete pipeline automatically:

```bash
python main.py
```

This will:
1. Check for existing vector database
2. If not found, run the complete pipeline (scrape, process, ingest)
3. Start the query interface

### Option 2: Manual Pipeline

Run each step manually for more control:

```bash
# 1. Scrape ReadTheDocs (openBIS)
python main.py scrape readthedocs --url https://openbis.readthedocs.io/en/latest/ --output data/raw/openbis

# 2. Scrape Wiki.js (configure URL first)
python main.py scrape wikijs --url https://your-wikijs-site.com --output data/raw/wikijs

# 3. Process all scraped content
python main.py process --input data/raw --output data/processed

# 4. Ingest into vector database
python main.py ingest --chunks-file data/processed/chunks.json --reset

# 5. Start query interface
python main.py query
```

## Configuration

### Wiki.js URL

Edit `src/desi/__main__.py` and update the `DEFAULT_WIKIJS_URL` variable:

```python
DEFAULT_WIKIJS_URL = "https://your-wikijs-site.com"
```

### Other Settings

You can customize various settings:

- **Database path**: `--db-path desi_vectordb`
- **Collection name**: `--collection-name desi_docs`
- **Ollama model**: `--model qwen3`
- **Chunk sizes**: `--min-chunk-size 100 --max-chunk-size 1000`

## Usage

### Command Line Interface

```bash
# Start interactive query session
python main.py query

# Show database statistics
python main.py query --stats

# Use different model
python main.py query --model llama2
```

### Web Interface

```bash
# Start web interface
python -m desi.web.cli

# Custom host and port
python -m desi.web.cli --host 0.0.0.0 --port 8080
```

Then open your browser to `http://localhost:5000`

### Pipeline Scripts

```bash
# Run complete pipeline with custom settings
python scripts/run_pipeline.py --wikijs-url https://your-wiki.com --max-pages 100 --start-query

# Skip scraping, only process existing data
python scripts/run_pipeline.py --skip-scraping

# Reset database and re-ingest
python scripts/run_pipeline.py --skip-scraping --skip-processing --reset-db
```

## Example Queries

- "How do I create a new experiment in openBIS?"
- "How to log in to the BAM Data Store?"
- "What are the steps to register a collection?"
- "How do I upload data to the data store?"
- "What is the difference between spaces and projects in openBIS?"

## Source Prioritization

DeSi automatically prioritizes content based on the query context:

- **Data Store operations**: Prioritizes Wiki.js content
- **General openBIS questions**: Uses both sources with balanced weighting
- **Specific technical queries**: Retrieves most relevant content regardless of source

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
├── data/               # Data storage (raw and processed)
├── requirements.txt    # Python dependencies
├── main.py            # Main entry point
└── README.md          # This file
```

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

## License

This project is part of the chatBIS ecosystem. Please refer to the main project license.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs with `--verbose` flag
3. Ensure all prerequisites are properly installed
4. Verify that Ollama models are available
