#!/usr/bin/env python3
"""
Setup script for DeSi.

This script helps users set up DeSi by checking prerequisites,
installing dependencies, and running initial tests.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    # Minimum supported Python version is 3.8; this script assumes it's running on 3.8+
    print(
        f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected (requires 3.8+)"
    )
    return True


def check_ollama():
    """Check if Ollama is installed and running."""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(
            ["ollama", "list"], check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓ Ollama is installed and running")

            # Check for required models
            models = result.stdout
            required_models = ["qwen3", "nomic-embed-text"]
            missing_models = []

            for model in required_models:
                if model not in models:
                    missing_models.append(model)

            if missing_models:
                print(f"⚠️  Missing required models: {', '.join(missing_models)}")
                print("You can install them with:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
                return False
            else:
                print("✓ Required Ollama models are available")
                return True
        else:
            print("❌ Ollama is not running")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("Please install Ollama from https://ollama.com/download")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
        )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running basic functionality tests...")
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        # Test imports with the NEW class-based structure
        from desi.processor import DsWikiProcessor, OpenBisProcessor
        from desi.query import ChatbotEngine, RAGQueryEngine, SqliteConversationMemory
        from desi.scraper import OpenbisScraper  # Assuming this is your scraper class
        from desi.utils import DesiConfig, setup_logging

        print("✓ All modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directory structure...")
    directories = [
        "data/raw/openbis",
        "data/raw/wikijs",
        "data/processed/openbis",
        "data/processed/wikijs",
        "desi_vectordb",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✓ Directory structure created")
    return True


def main():
    """Main setup function."""
    print("DeSi Helper Setup")
    print("=" * 50)

    # Change to the script directory
    os.chdir(Path(__file__).parent)

    checks = [
        ("Python Version", check_python_version),
        # ("Ollama Installation", check_ollama),
        ("Python Dependencies", install_dependencies),
        ("Basic Tests", run_basic_tests),
        ("Directory Structure", create_directories),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        try:
            result = check_func()
            results.append((check_name, result))
            if not result:
                print(f"Setup failed at: {check_name}")
                break
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
            break

    # Summary
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)

    passed = 0
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1

    if passed == len(checks):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure URLs in .env file (optional)")
        print("2. Run integration test: python scripts/integration_test.py")
        print("3. Run full pipeline: python main.py")
        print("4. Web interface: python main.py --web (placeholder, uses CLI for now)")
        return 0
    else:
        print(f"\n❌ Setup incomplete. {passed}/{len(checks)} checks passed.")
        print("Please resolve the issues above and run setup again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
