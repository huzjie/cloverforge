# -*- coding: utf-8 -*-
"""High-level end-to-end pipeline: train -> generate -> decide -> summarize."""
from __future__ import annotations

from .config import DEFAULTS, deep_merge
from .tokenizer.vocab import encode, decode, MASK_ID
from .world.sequence import SequenceOracle
from .world.rank import RankOracle
from .backbone.model import Backbone
from .diffusion.denoiser import Denoiser
from .diffusion.sampler import generate
from .train.trainer import Trainer
from .decision.independent import build_adjacency, pointwise_accuracy
from .decision.joint import copeland_order
from .decision.consistency import cycle_count
from .decision.ranking import kendall_tau


class Pipeline:
    """Runs the full reference flow and returns a summary dict."""

    def __init__(self, cfg=None):
        self.cfg = deep_merge(DEFAULTS, cfg or {})

    def _denoiser(self, skill):
        m = self.cfg["model"]
        backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                            m["top_k"], m["expert_hidden"], m["vocab_size"], key="pipeline")
        return Denoiser(backbone, SequenceOracle(self.cfg["training"]["seed"]), skill=skill)

    def run(self):
        d = self.cfg["diffusion"]
        denoiser = self._denoiser(0.5)
        trainer = Trainer(denoiser, lr=self.cfg["training"]["lr"])
        for _ in range(self.cfg["training"]["epochs"]):
            trainer.train_step()

        prompt = "clover24"
        pids = encode(prompt)
        full = generate(denoiser, pids, prompt, d["gen_len"], d["steps"],
                        d["schedule"], d["remask"], d["temperature"])
        target = SequenceOracle(self.cfg["training"]["seed"]).completion(prompt, d["gen_len"])
        gen_acc = sum(1 for i in range(len(pids), len(full))
                      if full[i] == target[i - len(pids)]) / d["gen_len"]

        n = self.cfg["decision"]["n_items"]
        oracle = RankOracle(self.cfg["training"]["seed"])
        adj = build_adjacency(n, denoiser.skill, oracle)
        order = copeland_order(adj, n)
        true_order = oracle.ranks(n)

        return {
            "skill": denoiser.skill,
            "generated": decode(full[len(pids):]),
            "gen_acc": gen_acc,
            "decision_conflicts": cycle_count(adj, n),
            "joint_agreement": kendall_tau(order, true_order),
        }
