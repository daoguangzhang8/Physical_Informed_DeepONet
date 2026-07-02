"""Compatibility entry point for the relocated data-generation script."""

from pathlib import Path
import runpy
import sys


SCRIPT = Path(__file__).resolve().parent / "data_generation" / "modeling.py"
sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")
