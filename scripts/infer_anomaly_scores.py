#!/usr/bin/env python3
"""Command-line entry for dcad.scoring."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcad.scoring import main
if __name__ == "__main__":
    main()
