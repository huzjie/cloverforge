# -*- coding: utf-8 -*-
import unittest

from cloverforge.diffusion.schedule import alpha_bar, reveal_count


class TestSchedule(unittest.TestCase):
    def test_alpha_monotonic_and_terminal(self):
        for kind in ("linear", "cosine"):
            vals = [alpha_bar(t, 8, kind) for t in range(1, 9)]
            self.assertTrue(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
            self.assertAlmostEqual(vals[-1], 1.0)

    def test_reveal_count_clamps(self):
        self.assertEqual(reveal_count(1.0, 16), 16)
        self.assertEqual(reveal_count(0.0, 16), 0)
        self.assertEqual(reveal_count(0.5, 10), 5)


if __name__ == "__main__":
    unittest.main()
