# -*- coding: utf-8 -*-
"""Full denoiser backbone: embedding -> blocks -> LM head."""
from __future__ import annotations

from .embedding import Embedding
from .block import Block
from .lm_head import LMHead


class Backbone:
    def __init__(self, dim, n_layers, n_heads, n_experts, top_k, expert_hidden,
                 vocab_size, key="cloverforge"):
        self.embedding = Embedding(vocab_size, dim, key)
        self.blocks = [Block(dim, n_heads, n_experts, top_k, expert_hidden,
                             "%s:b%d" % (key, i)) for i in range(n_layers)]
        self.lm_head = LMHead(dim, vocab_size, key)

    def forward(self, token_ids, skill=0.5):
        """token_ids: [L]; returns logits [L, vocab_size]."""
        X = [self.embedding.embed(t) for t in token_ids]
        for block in self.blocks:
            X = block.forward(X, skill)
        return [self.lm_head.forward(h) for h in X]
