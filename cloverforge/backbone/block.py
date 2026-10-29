# -*- coding: utf-8 -*-
"""A single backbone block: attention residual + sparse-MoE residual."""
from __future__ import annotations

from ..utils.linalg import add
from .attention import MultiHeadAttention
from ..moe.gate import MoEBlock


class Block:
    def __init__(self, dim, n_heads, n_experts, top_k, expert_hidden, key):
        self.attn = MultiHeadAttention(dim, n_heads, "%s:attn" % key)
        self.moe = MoEBlock(dim, n_experts, top_k, expert_hidden, "%s:moe" % key)

    def forward(self, X, skill):
        # X: [L, dim]
        A = self.attn.forward(X)
        H = add(X, A)
        M = [self.moe.forward(h, skill, "tok:%d" % i) for i, h in enumerate(H)]
        return add(H, M)
