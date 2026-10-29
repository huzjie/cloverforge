# -*- coding: utf-8 -*-
"""A tiny synthetic symbol vocabulary -- self-contained, no training, no assets.

60 printable symbols (digits + lowercase + a subset of uppercase) plus 4
special tokens. This keeps the reference implementation dependency-free while
still exercising the exact same token-id mechanics a byte-level BPE would.
"""
from __future__ import annotations

PAD_ID = 0
MASK_ID = 1
BOS_ID = 2
EOS_ID = 3
CONTENT_OFFSET = 4

# 10 digits + 26 lowercase + 24 uppercase == 60 symbols
SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX"
N_SYMBOLS = len(SYMBOLS)
VOCAB_SIZE = CONTENT_OFFSET + N_SYMBOLS

_SPECIAL = {PAD_ID: "<pad>", MASK_ID: "<mask>", BOS_ID: "<bos>", EOS_ID: "<eos>"}


def symbol_to_id(symbol: str) -> int:
    idx = SYMBOLS.index(symbol)
    return CONTENT_OFFSET + idx


def id_to_symbol(token_id: int) -> str:
    if token_id in _SPECIAL:
        return _SPECIAL[token_id]
    return SYMBOLS[token_id - CONTENT_OFFSET]


def encode(text: str) -> list:
    """Encode a string into token ids (unknown chars become the mask token)."""
    out = []
    for ch in text:
        if ch in SYMBOLS:
            out.append(symbol_to_id(ch))
        else:
            out.append(MASK_ID)
    return out


def decode(ids: list) -> str:
    return "".join(id_to_symbol(t) for t in ids)
