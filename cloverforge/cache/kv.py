# -*- coding: utf-8 -*-
"""A minimal position-indexed KV cache.

Used by semi-autoregressive decoding: earlier chunks compute their key/value
once and cache them so later chunks attend to them without recomputation.
"""
from __future__ import annotations


class KVCache:
    def __init__(self):
        self.keys = {}
        self.values = {}
        self.count = 0

    def put(self, pos: int, key: list, value: list):
        self.keys[pos] = key
        self.values[pos] = value
        self.count += 1

    def get(self, pos: int):
        return self.keys.get(pos), self.values.get(pos)

    def snapshot(self):
        return [(p, self.keys[p], self.values[p]) for p in sorted(self.keys)]

    def clear(self):
        self.keys.clear()
        self.values.clear()
        self.count = 0
