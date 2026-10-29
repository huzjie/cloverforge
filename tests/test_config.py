# -*- coding: utf-8 -*-
import unittest

from cloverforge.config import load_config, deep_merge, DEFAULTS


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config()
        self.assertIn("model", cfg)
        self.assertIn("diffusion", cfg)

    def test_deep_merge(self):
        a = {"x": {"y": 1, "z": 2}}
        b = {"x": {"y": 9}}
        out = deep_merge(a, b)
        self.assertEqual(out["x"]["y"], 9)
        self.assertEqual(out["x"]["z"], 2)


if __name__ == "__main__":
    unittest.main()
