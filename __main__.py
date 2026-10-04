"""Enables `python -m apibench`."""
import os
import sys

try:
    from .apibench import main
except ImportError:  # running the file directly: python apibench/__main__.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apibench import main

sys.exit(main())
