# muon-minimal

A minimal PyTorch implementation of the **Muon** optimizer, including the two
changes that made it work at LLM scale.

> **Muon is Scalable for LLM Training**
> Liu et al., Moonshot AI — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
> (Muon itself is by Keller Jordan et al., 2024)

Muon is SGD-with-momentum where the momentum buffer is replaced by its nearest
semi-orthogonal matrix before the step:

```
M_t = mu * M_{t-1} + G_t
O_t = NewtonSchulz(mu * M_t + G_t)        # Nesterov variant
W_t = W_{t-1} - lr * (0.2 * sqrt(max(A,B)) * O_t + lambda * W_{t-1})
```

Orthogonalizing means every direction in the update gets comparable magnitude,
so anisotropic curvature stops dictating the step size. The orthogonal factor
is approximated with 5 iterations of the quintic `f(x) = 3.4445x - 4.7750x³ +
2.0315x⁵` in bfloat16 — deliberately *inexact*, because Muon does not need
exact orthogonality and the iteration is the per-step cost.

Moonshot's two additions:

1. **decoupled weight decay** — without it, weights grow unbounded over a long run
2. **consistent update RMS** — an orthogonalized `[A, B]` update has RMS
   `1/sqrt(max(A,B))`, so scaling by `0.2 * sqrt(max(A,B))` pins it at 0.2 for
   every shape. That lands in AdamW's usual 0.2–0.4 band, which is why **you can
   reuse your AdamW learning rate and weight decay directly.**

## Install

```bash
pip install torch
```

## Use

Muon takes 2D hidden weights. Embeddings, the LM head, norms and biases stay on
AdamW — shape alone can't tell those apart from a hidden matrix, so routing is
by name:

```python
from muon import Muon, split_params

muon_params, adam_params = split_params(model)
opts = [
    Muon(muon_params, lr=3e-3, momentum=0.95, weight_decay=0.01),
    torch.optim.AdamW(adam_params, lr=3e-3, weight_decay=0.01),
]

for step in range(steps):
    loss = model(*batch())
    for o in opts: o.zero_grad()
    loss.backward()
    for o in opts: o.step()
```

`split_params` sends a 2D weight to AdamW if its name contains any of
`embed, wte, wpe, lm_head, norm, ln, bias`; pass `extra_adam_hints=(...)` for
your own naming.

## Verify

```bash
python test_muon.py   # 9 checks, a few seconds
python bench.py       # tiny transformer, Muon+AdamW vs AdamW, ~30s
```

What the checks pin down:

- **the quintic works** — condition number collapses from >100 to <2 in 5 steps,
  on square, wide and tall matrices
- **it is inexact, as intended** — singular values land in a band around
  [0.65, 1.2], not exactly 1; the test asserts the band, not perfection
- **direction is preserved** — an orthogonal input comes back parallel to itself
  (cosine > 0.99), scaled but not rotated
- **shape-consistent steps** — update RMS varies <1.3x across 5 shapes with the
  scale on, >2.5x with it off. This is the whole reason lr transfers.
- **the mechanism, isolated** — same lr, same momentum, orthogonalized vs not:
  Muon leaves 10–30x less loss than SGD+momentum across seeds
- weight decay, non-matrix rejection, and `split_params` routing

## Honest scope

`test_muon.py` deliberately does **not** assert "Muon beats AdamW". On a
full-batch toy, AdamW's `m/sqrt(v)` saturates near 1 and gives it a ~5x larger
effective step, so it wins. The paper's ~2x compute-efficiency result is a
claim about billion-parameter pretraining and a CPU test cannot settle it — so
`bench.py` prints both curves and asserts only that each one trains.

Not implemented: distributed/ZeRO-1 sharding of the Newton-Schulz work, which is
Section 3's other contribution and the reason Muon is practical on a real
cluster.

## License

MIT
