# -*- coding: utf-8 -*-
"""Example: skill-driven training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.tokenizer.vocab import encode, MASK_ID
from cloverforge.world.sequence import SequenceOracle
from cloverforge.backbone.model import Backbone
from cloverforge.diffusion.denoiser import Denoiser
from cloverforge.train.trainer import Trainer
from cloverforge.train.loss import masked_cross_entropy
from cloverforge.utils.linalg import argmax

cfg = load_config()
m, d = cfg["model"], cfg["diffusion"]
backbone = Backbone(m["dim"], m["n_layers"], m["n_heads"], m["n_experts"],
                    m["top_k"], m["expert_hidden"], m["vocab_size"], key="example")
denoiser = Denoiser(backbone, SequenceOracle(cfg["training"]["seed"]), skill=0.5)
trainer = Trainer(denoiser, lr=cfg["training"]["lr"])

prompt, pids = "clover24", encode("clover24")
target = SequenceOracle(cfg["training"]["seed"]).completion(prompt, d["gen_len"])
x = list(pids) + [MASK_ID] * d["gen_len"]

for ep in range(cfg["training"]["epochs"] + 1):
    lg = denoiser.logits(x, prompt, len(pids))
    loss = masked_cross_entropy(lg, x, target, len(pids))
    acc = sum(1 for i in range(len(pids), len(x)) if argmax(lg[i]) == target[i - len(pids)]) / d["gen_len"]
    print("epoch=%2d  skill=%.3f  recon_acc=%.3f  loss=%.4f" % (ep, denoiser.skill, acc, loss))
    if ep < cfg["training"]["epochs"]:
        trainer.train_step()
