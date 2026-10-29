# -*- coding: utf-8 -*-
import unittest

from cloverforge.eval.speedup import parallel_speedup, semi_ar_speedup, speedup_curve


class TestSpeedup(unittest.TestCase):
    def test_parallel_grows_with_length(self):
        self.assertGreater(parallel_speedup(32, 8), parallel_speedup(8, 8))

    def test_semi_ar_reduces_speedup(self):
        self.assertLess(semi_ar_speedup(32, 8, 4), parallel_speedup(32, 8))

    def test_curve(self):
        curve = speedup_curve(32, 8)
        self.assertEqual(curve[0][0], 8)
        self.assertEqual(curve[-1][0], 32)


if __name__ == "__main__":
    unittest.main()
