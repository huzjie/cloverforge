# -*- coding: utf-8 -*-
"""Stdlib HTTP serving: /health, /v1/models, /v1/generate, /v1/decision."""
from .api import handle_generate, handle_decision, handle_models
from .server import serve

__all__ = ["handle_generate", "handle_decision", "handle_models", "serve"]
