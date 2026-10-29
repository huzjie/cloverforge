# -*- coding: utf-8 -*-
import unittest

from cloverforge.train.loss import masked_cross_entropy
from cloverforge.tokenizer.vocab import MASK_ID, encode
from cloverforge.world.sequence import SequenceOracle
from cloverforge.backbone.model import Backbone
from cloverforge.diffusion.denoiser import Denoiser


class TestLoss(unittest.TestCase):
    def test_loss_decreases_with_skill(self):
        m = dict(dim=16, n_layers=1, n_heads=2, n_experts=2, top_k=1, expert_hidden=32, vocab_size=64)
        bb = Backbone(**m, key="t")
        o = SequenceOracle()
        prompt, pids = "ab", encode("ab")
        target = o.completion(prompt, 6)
        x = list(pids) + [MASK_ID] * 6
        losses = []
        for s in (0.5, 1.0):
            dn = Denoiser(bb, o, skill=s)
            lg = dn.logits(x, prompt, 2)
            losses.append(masked_cross_entropy(lg, x, target, 2))
        self.assertLess(losses[1], losses[0])


if __name__ == "__main__":
    unittest.main()
