# -*- coding: utf-8 -*-
import unittest

from cloverforge.moe.router import router_logits, noisy_topk
from cloverforge.moe.gate import MoEBlock
from cloverforge.moe.load_balance import load_balance_stats
from cloverforge.utils.rng import stable_vector


class TestRouter(unittest.TestCase):
    def test_topk_and_weights(self):
        logits = [0.1, 0.5, 0.3, 0.9]
        order, weights = noisy_topk(logits, 2, 0.0, "k")
        self.assertEqual(len(order), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_router_logits_shape(self):
        W = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        out = router_logits([1.0, 1.0], W)
        self.assertEqual(len(out), 3)


class TestLoadBalance(unittest.TestCase):
    def test_aux_decreases_with_skill(self):
        moe = MoEBlock(16, 4, 2, 32, "t")
        batch = [stable_vector("b:%d" % i, 16) for i in range(8)]
        auxs = []
        for s in (0.5, 1.0):
            gp = [moe.gate_probs(v, s) for v in batch]
            _, _, aux = load_balance_stats(gp)
            auxs.append(aux)
        self.assertLess(auxs[1], auxs[0])


if __name__ == "__main__":
    unittest.main()
