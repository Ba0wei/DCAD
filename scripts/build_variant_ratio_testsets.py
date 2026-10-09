#!/usr/bin/env python3
"""Command-line entry point for variant-ratio data pool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcad.variant_ratio.pool import main

if __name__ == "__main__":
    main()
