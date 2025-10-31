#!/usr/bin/env python3
"""
CLI launcher for DeSi web interface.

This module provides a command-line interface to launch the DeSi web server
using uvicorn with FastAPI and Gradio.
"""

import argparse
import logging
import sys

from ..utils.config import DesiConfig
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the web interface CLI."""
    parser = argparse.ArgumentParser(
        description="Launch DeSi web interface with FastAPI and Gradio"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 7860)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (.env)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = DesiConfig(args.config)

    # Use command-line args or fall back to config
    host = args.host or config.web_host
    port = args.port or config.web_port
    reload = args.reload or config.web_debug

    logger.info("=" * 60)
    logger.info("🚀 Starting DeSi Web Interface")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: {reload}")
    logger.info("=" * 60)

    try:
        import uvicorn

        # Update the global API base URL in the app module
        import desi.web.app as app_module

        app_module.api_base_url = f"http://{host}:{port}"

        # Run the server
        uvicorn.run(
            "desi.web.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )

    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        logger.error("Please install required packages: pip install fastapi uvicorn gradio")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start web server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

