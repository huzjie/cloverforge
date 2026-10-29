# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / train / generate / decision / serve."""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .tokenizer.vocab import encode, decode, MASK_ID
from .world.sequence import SequenceOracle
from .world.rank import RankOracle
from .backbone.model import Backbone
from .moe.gate import MoEBlock
from .moe.load_balance import load_balance_stats
from .diffusion.denoiser import Denoiser
from .diffusion.sampler import generate
from .decision.independent import build_adjacency, pointwise_accuracy
from .decision.joint import copeland_order
from .decision.consistency import cycle_count
from .decision.ranking import kendall_tau
from .utils.rng import stable_vector


def _denoiser(cfg, skill):
    m = cfg["model"]
    backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                        m["top_k"], m["expert_hidden"], m["vocab_size"], key="cli")
    return Denoiser(backbone, SequenceOracle(cfg["training"]["seed"]), skill=skill)


def cmd_doctor(cfg):
    m, d = cfg["model"], cfg["diffusion"]
    print("[doctor] backbone forward smoke ...", flush=True)
    denoiser = _denoiser(cfg, 0.5)
    prompt = "clover24"
    pids = encode(prompt)
    x = list(pids) + [MASK_ID] * d["gen_len"]
    logits = denoiser.logits(x, prompt, len(pids))
    assert len(logits) == len(x), "logits length mismatch"
    assert len(logits[0]) == m["vocab_size"], "vocab size mismatch"
    print("[doctor]   denoiser logits %dx%d OK" % (len(logits), len(logits[0])))

    from .utils.linalg import argmax
    target = SequenceOracle(cfg["training"]["seed"]).completion(prompt, d["gen_len"])
    acc05 = sum(1 for i in range(len(pids), len(x))
                if argmax(logits[i]) == target[i - len(pids)]) / d["gen_len"]
    denoiser.skill = 0.9
    logits = denoiser.logits(x, prompt, len(pids))
    acc09 = sum(1 for i in range(len(pids), len(x))
                if argmax(logits[i]) == target[i - len(pids)]) / d["gen_len"]
    print("[doctor]   recon acc skill=0.5 %.3f -> skill=0.9 %.3f" % (acc05, acc09))

    print("[doctor] MoE load balance smoke ...", flush=True)
    moe = MoEBlock(m["dim"], m["n_experts"], m["top_k"], m["expert_hidden"], "cli:moe")
    batch = [stable_vector("doctor:%d" % i, m["dim"]) for i in range(16)]
    for s in (0.5, 1.0):
        gp = [moe.gate_probs(v, s) for v in batch]
        _, _, aux = load_balance_stats(gp)
        print("[doctor]   moe aux skill=%.1f -> %.3f" % (s, aux))

    print("[doctor] diffusion generate smoke ...", flush=True)
    denoiser.skill = 0.95
    full = generate(denoiser, pids, prompt, d["gen_len"], d["steps"],
                    d["schedule"], d["remask"], d["temperature"])
    print("[doctor]   prompt=%r -> %r" % (prompt, decode(full[len(pids):])))

    print("[doctor] joint-decision smoke ...", flush=True)
    n_items = cfg["decision"]["n_items"]
    oracle = RankOracle(cfg["training"]["seed"])
    true_order = oracle.ranks(n_items)
    adj = build_adjacency(n_items, 0.8, oracle)
    cyc = cycle_count(adj, n_items)
    order = copeland_order(adj, n_items)
    tau = kendall_tau(order, true_order)
    print("[doctor]   independent conflicts=%d  joint agreement=%.3f" % (cyc, (1.0 + tau) / 2.0))
    print("[doctor] ALL OK", flush=True)
    return 0


def cmd_bench(cfg):
    from .eval.benchmark import run_all, summarize
    res = run_all(cfg)
    text, checks = summarize(res)
    print(text, flush=True)
    return 0 if all(checks.values()) else 1


def cmd_train(cfg):
    from .train.trainer import Trainer
    from .train.loss import masked_cross_entropy
    from .utils.linalg import argmax
    d = cfg["diffusion"]
    denoiser = _denoiser(cfg, 0.5)
    prompt = "clover24"
    pids = encode(prompt)
    target = SequenceOracle(cfg["training"]["seed"]).completion(prompt, d["gen_len"])
    x = list(pids) + [MASK_ID] * d["gen_len"]
    trainer = Trainer(denoiser, lr=cfg["training"]["lr"])
    print("[train] epoch  skill  recon_acc  loss", flush=True)
    for ep in range(cfg["training"]["epochs"] + 1):
        logits = denoiser.logits(x, prompt, len(pids))
        loss = masked_cross_entropy(logits, x, target, len(pids))
        acc = sum(1 for i in range(len(pids), len(x))
                  if argmax(logits[i]) == target[i - len(pids)]) / d["gen_len"]
        print("[train] %5d  %.3f   %.3f      %.4f" % (ep, denoiser.skill, acc, loss), flush=True)
        if ep < cfg["training"]["epochs"]:
            trainer.train_step()
    return 0


def cmd_generate(cfg, args):
    d = cfg["diffusion"]
    denoiser = _denoiser(cfg, args.skill)
    prompt = args.prompt
    pids = encode(prompt)
    steps = args.steps or d["steps"]
    full = generate(denoiser, pids, prompt, d["gen_len"], steps,
                    d["schedule"], d["remask"], d["temperature"])
    print("prompt   : %s" % prompt)
    print("generated: %s" % decode(full[len(pids):]))
    print("steps    : %d (parallel refinement)" % steps)
    return 0


def cmd_decision(cfg, args):
    n_items = cfg["decision"]["n_items"]
    oracle = RankOracle(cfg["training"]["seed"])
    true_order = oracle.ranks(n_items)
    adj = build_adjacency(n_items, args.skill, oracle)
    cyc = cycle_count(adj, n_items)
    ind_acc = pointwise_accuracy(n_items, args.skill, oracle)
    order = copeland_order(adj, n_items)
    tau = kendall_tau(order, true_order)
    print("independent mode: pointwise acc=%.3f  conflicts=%d" % (ind_acc, cyc))
    print("joint mode      : agreement=%.3f (tau=%.3f)  conflicts=0" % ((1.0 + tau) / 2.0, tau))
    print("joint order     : %s" % order)
    return 0


def cmd_speedup(cfg):
    from .eval.speedup import speedup_curve
    d = cfg["diffusion"]
    for L, sp in speedup_curve(80, d["steps"]):
        print("gen_len=%3d -> %.1fx speedup" % (L, sp))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="cloverforge",
                                description="diffusion language model (dLLM) reference implementation")
    p.add_argument("-c", "--config", default=None, help="recipe yaml file")
    p.add_argument("command", choices=["doctor", "bench", "train", "generate",
                                       "decision", "speedup", "serve", "version"])
    p.add_argument("--skill", type=float, default=0.9)
    p.add_argument("--prompt", default="clover24")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)

    if args.command == "version":
        from . import __version__
        print(__version__)
        return 0

    cfg = load_config(args.config)

    if args.command == "doctor":
        return cmd_doctor(cfg)
    if args.command == "bench":
        return cmd_bench(cfg)
    if args.command == "train":
        return cmd_train(cfg)
    if args.command == "generate":
        return cmd_generate(cfg, args)
    if args.command == "decision":
        return cmd_decision(cfg, args)
    if args.command == "speedup":
        return cmd_speedup(cfg)
    if args.command == "serve":
        from .serving.server import serve
        host = args.host or cfg["serving"]["host"]
        port = args.port or cfg["serving"]["port"]
        serve(host, port, cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
