# -*- coding: utf-8 -*-
"""Example: parallel diffusion text generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.tokenizer.vocab import encode, decode
from cloverforge.world.sequence import SequenceOracle
from cloverforge.backbone.model import Backbone
from cloverforge.diffusion.denoiser import Denoiser
from cloverforge.diffusion.sampler import generate

cfg = load_config()
m, d = cfg["model"], cfg["diffusion"]
backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                    m["top_k"], m["expert_hidden"], m["vocab_size"], key="example")
denoiser = Denoiser(backbone, SequenceOracle(cfg["training"]["seed"]), skill=0.95)

prompt = "clover24"
pids = encode(prompt)
full = generate(denoiser, pids, prompt, d["gen_len"], d["steps"],
                d["schedule"], d["remask"], d["temperature"])
print("prompt   :", prompt)
print("generated:", decode(full[len(pids):]))
print("(all %d tokens refined in %d parallel steps -- not one-by-one)" % (d["gen_len"], d["steps"]))
