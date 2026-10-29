# -*- coding: utf-8 -*-
import unittest

from cloverforge.diffusion.confidence import probs, confidence, predict_token, top_confident


class TestConfidence(unittest.TestCase):
    def test_probs(self):
        p = probs([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0)

    def test_predict(self):
        self.assertEqual(predict_token([1.0, 5.0, 3.0]), 1)

    def test_top_confident(self):
        m = {0: [0.1, 0.9], 1: [0.9, 0.1], 2: [0.5, 0.5]}
        top = top_confident(m, 1)
        self.assertEqual(len(top), 1)


if __name__ == "__main__":
    unittest.main()
