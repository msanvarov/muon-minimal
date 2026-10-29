"""Muon: momentum orthogonalised by Newton-Schulz, arXiv:2502.16982.

Muon (Keller Jordan) replaces the momentum buffer with its nearest
semi-orthogonal matrix before taking a step. Moonshot AI's "Muon is Scalable
for LLM Training" adds the two things it needed to work at LLM scale:

  1. decoupled weight decay, and
  2. a per-parameter update scale that makes Muon's update RMS match AdamW's,
     so an AdamW-tuned learning rate transfers directly.

Muon applies to 2D hidden weights. Embeddings, the LM head, norms and biases
stay on AdamW -- shape alone cannot tell you which is which, so use
`split_params` (or pass the groups yourself).
"""

import torch

# Quintic f(x) = ax + bx^3 + cx^5, coefficients tuned to push singular values
# toward 1 as fast as possible. Not a convergent iteration -- deliberately so.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate the orthogonal factor of G via `steps` quintic iterations.

    Runs in bfloat16: the iteration only has to land singular values *near* 1,
    so the paper calls this "inexact orthogonalization" and spends no more
    precision on it. N=5 is the paper's choice; N=10 buys accuracy but no
    downstream gain.
    """
    assert G.ndim == 2, f"expected a matrix, got shape {tuple(G.shape)}"
    a, b, c = NS_COEFFS
    X = G.bfloat16()
    tall = X.size(0) > X.size(1)
    if tall:  # the iteration is cheapest on wide matrices
        X = X.mT
    X = X / (X.norm() + eps)  # spectral norm <= Frobenius norm, so sigma <= 1
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.mT if tall else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters.

    Args:
        lr: reuse your AdamW learning rate -- that is the point of `rms_scale`.
        momentum / nesterov: Nesterov feeds `mu * buf + grad` to Newton-Schulz.
        ns_steps: Newton-Schulz iterations (5 in the paper).
        weight_decay: decoupled, applied to the parameter not the gradient.
        rms_scale: target update RMS. An orthogonalised [A, B] update has RMS
            1 / sqrt(max(A, B)), so scaling by `rms_scale * sqrt(max(A, B))`
            pins it at `rms_scale`. The paper uses 0.2, matching AdamW's
            observed 0.2-0.4.
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, rms_scale=0.2):
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("lr >= 0, 0 <= momentum < 1, weight_decay >= 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay,
                                      rms_scale=rms_scale))
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon takes 2D parameters; got {p.ndim}D {tuple(p.shape)}. "
                        "Put norms, biases, embeddings and the LM head on AdamW "
                        "(see split_params)."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = newtonschulz5(u, steps=g["ns_steps"])
                o = o * (g["rms_scale"] * max(p.shape) ** 0.5)
                if g["weight_decay"]:
                    o = o.add(p, alpha=g["weight_decay"])
                p.add_(o, alpha=-g["lr"])
        return loss


ADAM_NAME_HINTS = ("embed", "wte", "wpe", "lm_head", "norm", "ln", "bias")


def split_params(model, extra_adam_hints=()):
    """Split a model into (muon_params, adam_params).

    2D weights go to Muon unless their name looks like an embedding, the LM
    head, or a norm -- the paper keeps those on AdamW because their rows are
    token- or feature-indexed, not a linear map whose spectrum you want flat.
    """
    hints = ADAM_NAME_HINTS + tuple(extra_adam_hints)
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lowered = name.lower()
        (muon if p.ndim == 2 and not any(h in lowered for h in hints) else adam).append(p)
    return muon, adam
