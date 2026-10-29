# -*- coding: utf-8 -*-
"""Semi-autoregressive block decoding: split the output into parallel chunks."""
from __future__ import annotations


def semi_ar_blocks(gen_len: int, blocks: int):
    """Partition the generated region into ``blocks`` contiguous chunks.

    Each chunk is denoised in parallel; chunks are consumed left-to-right so
    later chunks can condition on earlier ones (a speed/quality trade-off).
    """
    if blocks <= 1:
        return [list(range(gen_len))]
    chunk = (gen_len + blocks - 1) // blocks
    out = []
    for b in range(blocks):
        start = b * chunk
        end = min(gen_len, start + chunk)
        if start < gen_len:
            out.append(list(range(start, end)))
    return out
