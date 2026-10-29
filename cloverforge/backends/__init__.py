# -*- coding: utf-8 -*-
"""Compute-backend abstraction: pure-python default, swappable for accelerators."""
from .base import Backend
from .pure import PureBackend

__all__ = ["Backend", "PureBackend"]
