# -*- coding: utf-8 -*-
"""Export model config and a generated sample to JSON (for reproducibility)."""
from __future__ import annotations

import json

from .config import load_config
from .pipeline import Pipeline


def export_sample(path: str, recipe: str | None = None):
    cfg = load_config(recipe)
    out = Pipeline(cfg).run()
    payload = {
        "config": cfg,
        "sample": out,
        "note": "cloverforge reference run (deterministic)",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
