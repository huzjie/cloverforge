# -*- coding: utf-8 -*-
"""Tokenizer package: a tiny deterministic symbol vocabulary."""
from .vocab import (
    PAD_ID, MASK_ID, BOS_ID, EOS_ID, CONTENT_OFFSET, N_SYMBOLS, VOCAB_SIZE,
    encode, decode, id_to_symbol, symbol_to_id,
)

__all__ = [
    "PAD_ID", "MASK_ID", "BOS_ID", "EOS_ID", "CONTENT_OFFSET",
    "N_SYMBOLS", "VOCAB_SIZE", "encode", "decode", "id_to_symbol", "symbol_to_id",
]
