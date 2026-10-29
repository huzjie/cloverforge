# -*- coding: utf-8 -*-
"""Runtime configuration with layered defaults (code default < recipe yaml < cli)."""
from __future__ import annotations

import copy

from .utils.yamlish import yaml_loads

# Defaults are deliberately small so a reference run stays interactive
# (<10s on a laptop), while remaining large enough to exercise every code path.
DEFAULTS = {
    "model": {
        "dim": 32,             # residual stream width
        "n_layers": 2,         # backbone blocks (each with sparse MoE FFN)
        "n_heads": 4,          # attention heads
        "n_experts": 4,        # sparse MoE experts per block
        "top_k": 2,            # routed experts per token
        "expert_hidden": 64,   # expert FFN hidden width
        "ffn_mult": 2,         # dense fallback FFN multiplier
        "vocab_size": 64,      # token vocabulary (60 symbols + 4 special)
        "max_len": 32,         # max sequence length
    },
    "diffusion": {
        "steps": 8,            # reverse diffusion refinement steps (T)
        "schedule": "cosine",  # reveal schedule: linear | cosine
        "remask": "confidence",  # remasking policy: confidence | random
        "prompt_len": 8,       # observed prefix length (prompt tokens)
        "gen_len": 16,         # generated suffix length
        "temperature": 1.0,
    },
    "decision": {
        "n_items": 16,          # ranking items for joint-decision demo
        "mode": "joint",        # independent | joint
        "judge_top_k": 3,       # candidates considered per question
    },
    "training": {
        "lr": 0.25,
        "epochs": 12,
        "target_skill": 0.95,
        "seed": 20261029,
    },
    "serving": {"host": "127.0.0.1", "port": 8788},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(recipe: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if recipe:
        try:
            with open(recipe, "r", encoding="utf-8") as f:
                user = yaml_loads(f.read())
            cfg = deep_merge(cfg, user)
        except FileNotFoundError:
            raise SystemExit("[config] recipe not found: %s" % recipe)
    return cfg
