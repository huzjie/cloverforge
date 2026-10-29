# -*- coding: utf-8 -*-
import unittest

from cloverforge.config import load_config
from cloverforge.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def test_run(self):
        cfg = load_config()
        cfg["training"]["epochs"] = 3
        out = Pipeline(cfg).run()
        self.assertIn("gen_acc", out)
        self.assertGreaterEqual(out["skill"], 0.5)
        self.assertGreaterEqual(out["gen_acc"], 0.0)
        self.assertLessEqual(out["gen_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
