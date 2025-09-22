#!/usr/bin/env python3
"""
Main entry point for DeSi.

This script provides a simple way to run DeSi from the command line.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main module
from desi.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
