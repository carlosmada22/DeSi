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
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def check_ollama():
    """Check if Ollama is installed and running."""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            
            # Check for required models
            models = result.stdout
            required_models = ["gpt-oss:20b", "nomic-embed-text"]
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
        print("Please install Ollama from https://ollama.ai/")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
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
        
        # Test imports
        from desi.scraper.readthedocs_scraper import ReadTheDocsScraper
        from desi.processor.processor import MultiSourceRAGProcessor
        from desi.utils.vector_db import DesiVectorDB
        from desi.query.query import DesiRAGQueryEngine
        
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
        "data/processed"
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
        #("Ollama Installation", check_ollama),
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
        print("1. Configure Wiki.js URL in src/desi/__main__.py (optional)")
        print("2. Run integration test: python scripts/integration_test.py")
        print("3. Run full pipeline: python main.py")
        print("4. Or start with web interface: python main.py web")
        return 0
    else:
        print(f"\n❌ Setup incomplete. {passed}/{len(checks)} checks passed.")
        print("Please resolve the issues above and run setup again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
