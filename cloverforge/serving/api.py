# -*- coding: utf-8 -*-
"""JSON handlers for the HTTP API."""
from __future__ import annotations

from ..config import DEFAULTS, deep_merge
from ..tokenizer.vocab import encode, decode, MASK_ID
from ..world.sequence import SequenceOracle
from ..world.rank import RankOracle
from ..backbone.model import Backbone
from ..diffusion.denoiser import Denoiser
from ..diffusion.sampler import generate
from ..decision.independent import build_adjacency
from ..decision.joint import copeland_order
from ..decision.consistency import cycle_count
from ..decision.ranking import kendall_tau


def _denoiser(cfg, skill):
    m = cfg["model"]
    backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                        m["top_k"], m["expert_hidden"], m["vocab_size"], key="serve")
    oracle = SequenceOracle(cfg["training"]["seed"])
    return Denoiser(backbone, oracle, skill=skill)


def handle_generate(cfg, body):
    prompt = body.get("prompt", "clover24")
    skill = float(body.get("skill", 0.9))
    d = cfg["diffusion"]
    steps = int(body.get("steps", d["steps"]))
    gen_len = int(body.get("gen_len", d["gen_len"]))
    denoiser = _denoiser(cfg, skill)
    prompt_ids = encode(prompt)
    full = generate(denoiser, prompt_ids, prompt, gen_len, steps,
                    d["schedule"], d["remask"], d["temperature"])
    gen = full[len(prompt_ids):]
    return {
        "prompt": prompt,
        "generated": decode(gen),
        "tokens": gen,
        "skill": skill,
        "steps": steps,
    }


def handle_decision(cfg, body):
    n_items = int(body.get("n_items", cfg["decision"]["n_items"]))
    skill = float(body.get("skill", 0.8))
    mode = body.get("mode", cfg["decision"]["mode"])
    oracle = RankOracle(cfg["training"]["seed"])
    true_order = oracle.ranks(n_items)
    adj = build_adjacency(n_items, skill, oracle)
    cyc = cycle_count(adj, n_items)
    if mode == "joint":
        order = copeland_order(adj, n_items)
        return {"mode": "joint", "order": order, "conflicts": 0,
                "agreement": kendall_tau(order, true_order),
                "independent_conflicts": cyc}
    # independent mode: report the raw (possibly conflicting) adjacency
    return {"mode": "independent", "conflicts": cyc,
            "adjacency": [[1 if adj[i][j] else 0 for j in range(n_items)] for i in range(n_items)]}


def handle_models():
    return {"models": [{"id": "cloverforge-dllm", "object": "model",
                        "type": "diffusion-language-model"}]}
