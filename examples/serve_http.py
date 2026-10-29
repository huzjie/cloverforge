# -*- coding: utf-8 -*-
"""Example: start the HTTP server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.serving.server import serve

cfg = load_config()
serve(cfg["serving"]["host"], cfg["serving"]["port"], cfg)
