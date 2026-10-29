# -*- coding: utf-8 -*-
"""A single MoE block: routes a token to top-k experts and blends their outputs."""
from __future__ import annotations

from ..utils.linalg import add, scale
from ..utils.rng import stable_matrix
from .router import router_logits, noisy_topk
from .experts import ExpertFFN


class MoEBlock:
    def __init__(self, dim: int, n_experts: int, top_k: int, expert_hidden: int, key: str):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.Wg = stable_matrix("%s:Wg" % key, n_experts, dim)
        self.experts = [ExpertFFN(dim, expert_hidden, "%s:ex%d" % (key, j))
                        for j in range(n_experts)]
        # initial "hot expert" bias: before load-balance training, expert 0 is
        # over-favoured; the auxiliary loss shrinks this as skill -> 1.
        self.hot_bias = 4.0

    def gate_logits(self, x: list, skill: float) -> list:
        boost = (skill - 0.5) * 2.0
        base = router_logits(x, self.Wg)
        hot = [self.hot_bias if j == 0 else 0.0 for j in range(self.n_experts)]
        logits = [base[j] + hot[j] for j in range(self.n_experts)]
        # Load-balancing training flattens the routing distribution: the
        # auxiliary loss effectively raises the gate "temperature", so a trained
        # router spreads load evenly across experts (monotonic in skill).
        temperature = 1.0 + boost * 8.0
        return [v / temperature for v in logits]

    def gate_probs(self, x: list, skill: float) -> list:
        from .router import softmax_probs
        return softmax_probs(self.gate_logits(x, skill))

    def forward(self, x: list, skill: float, key: str = "tok") -> list:
        logits = self.gate_logits(x, skill)
        idx, weights = noisy_topk(logits, self.top_k, noise_scale=0.1, key=key)
        out = [0.0] * self.dim
        for k in range(self.top_k):
            e = self.experts[idx[k]].forward(x)
            out = add(out, scale(e, weights[k]))
        return out
