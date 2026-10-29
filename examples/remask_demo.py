# -*- coding: utf-8 -*-
"""Example: confidence remasking schedule."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.diffusion.schedule import alpha_bar, reveal_count

steps = 8
length = 16
print("cosine reveal schedule (how many of %d tokens are revealed per step):" % length)
prev = 0
for t in range(1, steps + 1):
    n = reveal_count(alpha_bar(t, steps, "cosine"), length)
    print("  step %d: reveal %d (cumulative %d)" % (t, max(0, n - prev), n))
    prev = n
print("\nHigh-confidence tokens are revealed first; low-confidence stay masked.")
