# -*- coding: utf-8 -*-
"""Example: decode-step speedup curve."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.eval.speedup import speedup_curve, semi_ar_speedup

cfg = load_config()
steps = cfg["diffusion"]["steps"]
print("autoregressive steps = sequence length; dLLM steps = fixed %d" % steps)
for L, sp in speedup_curve(80, steps):
    print("  length=%3d -> %.1fx" % (L, sp))
print("semi-autoregressive (4 blocks) at length 64 -> %.1fx"
      % semi_ar_speedup(64, steps, 4))
