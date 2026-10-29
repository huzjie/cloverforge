# -*- coding: utf-8 -*-
"""The full benchmark: four falsifiable dimensions, all monotonic in skill."""
from __future__ import annotations

from ..config import DEFAULTS, deep_merge
from ..tokenizer.vocab import MASK_ID, encode
from ..world.sequence import SequenceOracle
from ..world.rank import RankOracle
from ..backbone.model import Backbone
from ..moe.gate import MoEBlock
from ..moe.load_balance import load_balance_stats
from ..diffusion.denoiser import Denoiser
from ..diffusion.sampler import generate
from ..utils.linalg import argmax, softmax
from ..utils.rng import stable_vector, stable_int
from ..decision.independent import build_adjacency, pointwise_accuracy
from ..decision.joint import copeland_order
from ..decision.consistency import cycle_count
from ..decision.ranking import kendall_tau
from ..train.data import make_prompt
from .metrics import token_accuracy
from .speedup import parallel_speedup

SKILL_POINTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _build(cfg):
    m = cfg["model"]
    backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                        m["top_k"], m["expert_hidden"], m["vocab_size"], key="bench")
    oracle = SequenceOracle(cfg["training"]["seed"])
    return backbone, oracle


def run_all(cfg=None, skill_points=None):
    cfg = deep_merge(DEFAULTS, cfg or {})
    m, d, dec = cfg["model"], cfg["diffusion"], cfg["decision"]
    pts = skill_points or SKILL_POINTS
    res = {
        "skill": list(pts),
        "recon_acc": [], "recon_ppl": [],
        "gen_acc": [],
        "moe_aux": [], "moe_maxfrac": [],
        "ind_cycle": [], "ind_acc": [], "joint_tau": [], "joint_acc": [],
    }

    backbone, oracle = _build(cfg)
    denoiser = Denoiser(backbone, oracle)
    prompt = make_prompt(d["prompt_len"], cfg["training"]["seed"])
    prompt_ids = encode(prompt)
    target = oracle.completion(prompt, d["gen_len"])

    # single MoE block for load-balance measurement
    moe = MoEBlock(m["dim"], m["n_experts"], m["top_k"], m["expert_hidden"], "bench:moe")
    batch = [stable_vector("bench:tok:%d" % i, m["dim"]) for i in range(16)]

    rank_oracle = RankOracle(cfg["training"]["seed"])
    n_items = dec["n_items"]
    true_order = rank_oracle.ranks(n_items)

    for skill in pts:
        denoiser.skill = skill

        # 1) one-step reconstruction of a fully-masked generated region
        x = list(prompt_ids) + [MASK_ID] * d["gen_len"]
        logits = denoiser.logits(x, prompt, d["prompt_len"])
        pred = [argmax(logits[i]) for i in range(d["prompt_len"], len(x))]
        res["recon_acc"].append(token_accuracy(pred, target))
        from ..train.loss import masked_cross_entropy
        loss = masked_cross_entropy(logits, x, target, d["prompt_len"])
        res["recon_ppl"].append(_ppl(loss))

        # 2) full multi-step diffusion generation
        full = generate(denoiser, prompt_ids, prompt, d["gen_len"], d["steps"],
                        d["schedule"], d["remask"], d["temperature"])
        res["gen_acc"].append(token_accuracy(full[d["prompt_len"]:], target))

        # 3) MoE load balance
        gate_probs = [moe.gate_probs(v, skill) for v in batch]
        fractions, _probs, aux = load_balance_stats(gate_probs)
        res["moe_aux"].append(aux)
        res["moe_maxfrac"].append(max(fractions))

        # 4) joint-decision: independent vs global coordination
        adj = build_adjacency(n_items, skill, rank_oracle)
        res["ind_cycle"].append(cycle_count(adj, n_items))
        res["ind_acc"].append(pointwise_accuracy(n_items, skill, rank_oracle))
        order = copeland_order(adj, n_items)
        tau = kendall_tau(order, true_order)
        res["joint_tau"].append(tau)
        res["joint_acc"].append((1.0 + tau) / 2.0)

    # 5) decode-step speedup (skill-independent, pure algorithm)
    res["speedup"] = parallel_speedup(d["gen_len"], d["steps"])
    res["gen_len"] = d["gen_len"]
    res["steps"] = d["steps"]
    return res


def _ppl(loss):
    import math
    return math.exp(min(loss, 100.0))


def _monotonic(values, decreasing=False):
    if len(values) < 2:
        return True
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if decreasing and b - a > 1e-9:
            return False
        if not decreasing and a - b > 1e-9:
            return False
    return True


def summarize(res):
    header = "skill   recon_acc  recon_ppl  gen_acc   moe_aux  maxfrac  ind_acc  joint_acc  ind_cyc"
    lines = [header]
    n = len(res["skill"])
    for i in range(n):
        lines.append(
            "%.2f   %.3f      %.2f      %.3f    %.3f    %.3f    %.3f    %.3f       %5d" % (
                res["skill"][i], res["recon_acc"][i], res["recon_ppl"][i],
                res["gen_acc"][i], res["moe_aux"][i], res["moe_maxfrac"][i],
                res["ind_acc"][i], res["joint_acc"][i], res["ind_cycle"][i],
            ))
    checks = {
        "recon_acc (up)": _monotonic(res["recon_acc"]),
        "recon_ppl (down)": _monotonic(res["recon_ppl"], decreasing=True),
        "gen_acc (up)": _monotonic(res["gen_acc"]),
        "moe_aux (down)": _monotonic(res["moe_aux"], decreasing=True),
        "moe_maxfrac (down)": _monotonic(res["moe_maxfrac"], decreasing=True),
        "ind_acc (up)": _monotonic(res["ind_acc"]),
        "joint_acc (up)": _monotonic(res["joint_acc"]),
    }
    lines.append("")
    lines.append("monotonic checks:")
    for k, v in checks.items():
        lines.append("  %-20s %s" % (k, "OK" if v else "FAIL"))
    lines.append("speedup = gen_len/steps = %d/%d = %.2fx" % (
        res["gen_len"], res["steps"], res["speedup"]))
    lines.append("note: ind_cyc = logical conflicts in independent mode (joint mode = 0 always)")
    return "\n".join(lines), checks
