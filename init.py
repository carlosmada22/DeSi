#!/usr/bin/env python3
"""
Setup script for DeSi.

This script helps users set up DeSi by checking prerequisites,
installing dependencies, and running initial tests.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import requests


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    # Minimum supported Python version is 3.8; this script assumes it's running on 3.8+
    logger.info(
        f"Python {sys.version_info.major}.{sys.version_info.minor} detected (requires 3.8+)"
    )
    return True


def check_ollama():
    """
    Check if the Ollama service is accessible and has the required models.
    """
    logger.info("Checking Ollama accessibility and models...")
    required_models = ["qwen3", "nomic-embed-text"]
    ollama_api_url = "http://localhost:11434/api/tags"

    try:
        response = requests.get(ollama_api_url, timeout=5)
        response.raise_for_status()
        logger.info("Ollama server is accessible.")

        data = response.json()
        # Get a list of the full model names, e.g., ['qwen3:latest', 'nomic-embed-text:latest']
        installed_models = [model["name"] for model in data.get("models", [])]

        missing_models = []
        for required in required_models:
            # Check if any installed model *starts with* the required name.
            # This correctly handles tags like ':latest', ':7b', etc.
            if not any(
                installed.startswith(required) for installed in installed_models
            ):
                missing_models.append(required)

        if not missing_models:
            logger.info("✓ Required Ollama models are available.")
            return True
        else:
            logger.info(f"Missing required models: {', '.join(missing_models)}")
            logger.info("You can install them with:")
            for model in missing_models:
                logger.info(f"  ollama pull {model}")
            return False

    except requests.exceptions.ConnectionError:
        logger.info("Ollama server is not accessible at http://localhost:11434.")
        logger.info("   Please ensure the Ollama application or service is running.")
        return False
    except requests.exceptions.RequestException as e:
        logger.info(f"Failed to query the Ollama API: {e}")
        return False


def run_basic_tests():
    """Run basic functionality tests."""
    logger.info("Running basic functionality tests...")
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        # Test imports with the NEW class-based structure
        from desi.processor import DsWikiProcessor, OpenBisProcessor
        from desi.query import ChatbotEngine, RAGQueryEngine, SqliteConversationMemory
        from desi.scraper import OpenbisScraper
        from desi.utils import DesiConfig, setup_logging

        logger.info("✓ All modules import successfully")
        return True
    except ImportError as e:
        logger.info(f"Import error: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    logger.info("Creating directory structure...")
    directories = [
        "data/raw/openbis",
        "data/raw/wikijs",
        "data/processed/openbis",
        "data/processed/wikijs",
        "desi_vectordb",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("✓ Directory structure created")
    return True


def create_env_file() -> bool:
    """
    Creates a .env file from .env.example if it doesn't already exist.
    """
    logger.info("Checking for .env file...")
    source_file = Path(".env.example")
    dest_file = Path(".env")

    # Safety check: Do not overwrite an existing .env file.
    if dest_file.exists():
        logger.info("✓ .env file already exists. Skipping creation.")
        return True

    # Check if the template file exists before trying to copy.
    if not source_file.exists():
        logger.info("Warning: .env.example file not found. Cannot create .env.")
        logger.info("   Please ensure .env.example is in the project root.")
        return False  # This step failed.

    try:
        shutil.copyfile(source_file, dest_file)
        logger.info(f"Created '{dest_file}' from '{source_file}'.")
        logger.info(
            "   You can now customize the variables in the .env file if needed."
        )
        return True
    except Exception as e:
        logger.info(f"Failed to create .env file: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("DeSi Helper Setup")
    logger.info("=" * 50)

    os.chdir(Path(__file__).parent)

    checks = [
        ("Python Version", check_python_version),
        ("Ollama", check_ollama),
        ("Basic Tests", run_basic_tests),
        ("Directory Structure", create_directories),
    ]
    # ... rest of the main function is unchanged ...
    results = []
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            result = check_func()
            results.append((check_name, result))
            if not result:
                logger.info(f"Setup failed at: {check_name}")
                break
        except Exception as e:
            logger.info(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
            break

    logger.info("\n" + "=" * 50)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 50)

    passed = 0
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{check_name}: {status}")
        if result:
            passed += 1

    if passed == len(checks):
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Configure URLs in .env file (optional)")
        logger.info("2. Run integration test: python scripts/integration_test.py")
        logger.info("3. Run full pipeline: python main.py")
    else:
        logger.info(f"\n❌ Setup incomplete. {passed}/{len(checks)} checks passed.")
        logger.info("Please resolve the issues above and run setup again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
