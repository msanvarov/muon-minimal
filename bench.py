"""Muon + AdamW hybrid vs pure AdamW on a tiny char-level transformer (CPU).

This is the realistic *usage* demo, not evidence. The paper's headline -- ~2x
the compute efficiency of AdamW -- is measured on billion-parameter pretraining
runs. A 30-second CPU run cannot confirm or refute that, and the numbers below
move with the seed. Read it as "here is how to wire Muon into a real model".
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from muon import Muon, split_params

TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
    "sphinx of black quartz, judge my vow. "
) * 64
VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}
DATA = torch.tensor([STOI[c] for c in TEXT])


class Tiny(nn.Module):
    def __init__(self, dim=96, heads=4, layers=2, ctx=64):
        super().__init__()
        self.ctx, self.heads, self.dh = ctx, heads, dim // heads
        self.embed = nn.Embedding(len(VOCAB), dim)
        self.pos_embed = nn.Embedding(ctx, dim)
        self.qkv = nn.ModuleList(nn.Linear(dim, 3 * dim, bias=False) for _ in range(layers))
        self.proj = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim, bias=False), nn.GELU(),
                          nn.Linear(4 * dim, dim, bias=False))
            for _ in range(layers)
        )
        self.norm1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.norm_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, len(VOCAB), bias=False)

    def forward(self, idx, targets):
        B, T = idx.shape
        h = self.embed(idx) + self.pos_embed(torch.arange(T))
        for i in range(len(self.proj)):
            q, k, v = self.qkv[i](self.norm1[i](h)).chunk(3, -1)
            q, k, v = (t.view(B, T, self.heads, self.dh).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + self.proj[i](a.transpose(1, 2).reshape(B, T, -1))
            h = h + self.mlp[i](self.norm2[i](h))
        logits = self.lm_head(self.norm_f(h))
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def batch(bs, ctx):
    i = torch.randint(0, len(DATA) - ctx - 1, (bs,))
    s = torch.stack([DATA[j:j + ctx] for j in i])
    t = torch.stack([DATA[j + 1:j + ctx + 1] for j in i])
    return s, t


def run(which, steps, lr, seed):
    torch.manual_seed(seed)
    model = Tiny()
    if which == "muon":
        muon_p, adam_p = split_params(model)
        opts = [Muon(muon_p, lr=lr, momentum=0.95, weight_decay=0.01),
                torch.optim.AdamW(adam_p, lr=lr, weight_decay=0.01)]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)]

    losses = []
    for step in range(1, steps + 1):
        x, y = batch(32, model.ctx)
        loss = model(x, y)
        for o in opts:
            o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  {which:>5}  step {step:4d}  loss {sum(losses[-20:]) / 20:.4f}")
    return sum(losses[-20:]) / 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3, help="same lr for both -- the point of rms_scale")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"tiny transformer, {len(VOCAB)}-token vocab, same lr={a.lr} for both\n")
    m = run("muon", a.steps, a.lr, a.seed)
    print()
    d = run("adamw", a.steps, a.lr, a.seed)
    print(f"\nfinal loss  muon+adamw {m:.4f}   adamw {d:.4f}")
    print("both must at least train; a winner at this scale means nothing")
    assert m < 2.0 and d < 2.0, (m, d)
    print("ok")


if __name__ == "__main__":
    main()
