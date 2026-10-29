# -*- coding: utf-8 -*-
import unittest

from cloverforge.cache.kv import KVCache


class TestKVCache(unittest.TestCase):
    def test_put_get(self):
        c = KVCache()
        c.put(0, [1.0], [2.0])
        k, v = c.get(0)
        self.assertEqual(k, [1.0])
        self.assertEqual(v, [2.0])
        self.assertEqual(c.count, 1)

    def test_clear(self):
        c = KVCache()
        c.put(0, [1.0], [2.0])
        c.clear()
        self.assertEqual(c.count, 0)


if __name__ == "__main__":
    unittest.main()
