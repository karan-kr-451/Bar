"""
Run from project root:
    python run.py

This file exists so Python's import system can find the `snapbar` package.
Do NOT run snapbar/main.py directly — imports will break.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snapbar.main import main

if __name__ == "__main__":
    main()
