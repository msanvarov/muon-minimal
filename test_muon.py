"""Self-checks for Muon. Run: python test_muon.py"""

import torch
import torch.nn as nn

from muon import Muon, newtonschulz5, split_params


def cond(m):
    sv = torch.linalg.svdvals(m.float())
    return (sv.max() / sv.clamp_min(1e-12).min()).item()


def test_newtonschulz_collapses_the_condition_number():
    """The point of the quintic: whatever goes in comes out nearly isotropic.

    Note it is *inexact* by design -- 5 steps park the singular values in a
    band around [0.65, 1.2] rather than driving them to exactly 1. Muon does
    not need exact orthogonality, and the paper says so explicitly.
    """
    torch.manual_seed(0)
    for shape in [(64, 64), (32, 128), (128, 32)]:
        u, _, vt = torch.linalg.svd(torch.randn(*shape), full_matrices=False)
        for decade in (-1, -2):
            g = (u * torch.logspace(0, decade, min(shape))) @ vt
            before, after = cond(g), cond(newtonschulz5(g))
            assert before > 9, (shape, before)
            assert after < 2.0, (shape, decade, before, after)
            assert after < before / 4, (shape, decade, before, after)


def test_newtonschulz_preserves_direction_on_orthogonal_input():
    """An orthogonal input comes back parallel to itself (scaled, not rotated).

    Frobenius normalization divides a d x d orthogonal matrix down to singular
    values 1/sqrt(d), and 5 steps only climb partway back -- so the magnitude
    moves but the subspace must not.
    """
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(48, 48))[0]
    out = newtonschulz5(q).float()
    scale = (out * q).sum() / q.pow(2).sum()
    assert scale > 0.5, scale.item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), q.flatten(), dim=0)
    assert cos > 0.99, cos.item()  # bf16 noise is the only thing left


def test_newtonschulz_sign_is_preserved():
    """Orthogonalization must not flip the update's direction."""
    torch.manual_seed(0)
    g = torch.randn(32, 64)
    assert (newtonschulz5(g).float() * g).sum() > 0


def update_rms(shape, rms_scale):
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(*shape))
    p.grad = torch.randn(*shape)
    before = p.detach().clone()
    Muon([p], lr=1.0, momentum=0.0, nesterov=False, rms_scale=rms_scale).step()
    return (before - p.detach()).pow(2).mean().sqrt().item()


def test_update_rms_is_consistent_across_shapes():
    """Moonshot's fix: 0.2 * sqrt(max(A,B)) makes update size shape-independent.

    Without it the update RMS is 1/sqrt(max(A,B)), so a 16x512 matrix moves ~3x
    less per step than a 64x64 one and no single learning rate suits both.
    """
    shapes = [(64, 64), (16, 512), (512, 16), (128, 256), (256, 128)]
    scaled = [update_rms(s, 0.2) for s in shapes]
    raw = [update_rms(s, None) if False else update_rms(s, 1.0) / max(s) ** 0.5
           for s in shapes]
    assert max(scaled) / min(scaled) < 1.3, dict(zip(map(str, shapes), scaled))
    assert all(0.12 < r < 0.22 for r in scaled), scaled
    # the unscaled update is what the fix is correcting for
    assert max(raw) / min(raw) > 2.5, dict(zip(map(str, shapes), raw))


def test_rejects_non_matrix_params():
    for bad in (torch.zeros(8), torch.zeros(2, 3, 4)):
        try:
            Muon([nn.Parameter(bad)])
        except ValueError as e:
            assert "2D" in str(e)
        else:
            raise AssertionError(f"accepted a {bad.ndim}D parameter")


def test_weight_decay_shrinks_an_unused_param():
    p = nn.Parameter(torch.ones(8, 8))
    p.grad = torch.zeros(8, 8)
    Muon([p], lr=0.1, weight_decay=0.5, momentum=0.0, nesterov=False).step()
    # zero grad -> orthogonalised update is zero, so only decay acts
    assert torch.allclose(p.detach(), torch.full((8, 8), 1 - 0.1 * 0.5), atol=1e-6)


def test_split_params_routes_by_name():
    model = nn.Sequential()
    model.add_module("embed", nn.Embedding(10, 8))
    model.add_module("fc", nn.Linear(8, 8))
    model.add_module("norm", nn.LayerNorm(8))
    model.add_module("lm_head", nn.Linear(8, 10, bias=False))
    muon, adam = split_params(model)
    assert len(muon) == 1 and muon[0].shape == (8, 8)
    assert sum(p.numel() for p in adam) == 10 * 8 + 8 + 8 + 8 + 10 * 8


def _train(opt_fn, decade=-2, steps=60, seed=0):
    """Deep linear net on an ill-conditioned target. Returns loss left, as a
    fraction of the loss at init, so runs are comparable across seeds."""
    torch.manual_seed(seed)
    d = 48
    u, _, vt = torch.linalg.svd(torch.randn(d, d))
    target = (u * torch.logspace(0, decade, d)) @ vt
    x = torch.randn(512, d)
    y = x @ target.T
    net = nn.Sequential(nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False))
    opt = opt_fn(list(net.parameters()))
    start = (net(x) - y).pow(2).mean().item()
    for _ in range(steps):
        loss = (net(x) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (net(x) - y).pow(2).mean().item() / start


def test_orthogonalization_beats_raw_momentum():
    """The mechanism, isolated: same momentum, same lr, orthogonalized or not.

    Muon is SGD-with-momentum plus Newton-Schulz, so this is the one comparison
    that attributes a gain to the orthogonalization itself. It is ~10-30x less
    remaining loss across seeds.

    (Deliberately not asserted: "Muon beats AdamW". On a full-batch toy like
    this AdamW's m/sqrt(v) saturates near 1, giving it a ~5x bigger effective
    step, and it wins. The paper's 2x compute-efficiency claim is about LLM
    pretraining at scale, which a CPU test cannot settle -- see bench.py.)
    """
    for seed in range(3):
        muon = _train(lambda p: Muon(p, lr=0.02, momentum=0.95), seed=seed)
        sgdm = _train(lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.95), seed=seed)
        assert muon < sgdm / 5, f"seed {seed}: muon {muon:.2e} vs sgdm {sgdm:.2e}"
        print(f"    seed {seed}: muon {muon:.2e} vs sgd+momentum {sgdm:.2e}")


def test_insensitive_to_target_conditioning():
    """Flattening the update spectrum should blunt the cost of bad conditioning."""
    for seed in range(2):
        easy = _train(lambda p: Muon(p, lr=0.02, momentum=0.95), decade=0, seed=seed)
        hard = _train(lambda p: Muon(p, lr=0.02, momentum=0.95), decade=-2, seed=seed)
        assert hard < 4 * easy, (seed, easy, hard)


def test_optimizer_converges_and_state_persists():
    p = nn.Parameter(torch.randn(16, 16))
    opt = Muon([p], lr=0.05, momentum=0.9)
    start = p.detach().pow(2).mean().item()
    for _ in range(50):
        loss = p.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert p.detach().pow(2).mean().item() < start * 0.5
    assert "momentum_buffer" in opt.state[p]


if __name__ == "__main__":
    for fn in [
        test_newtonschulz_collapses_the_condition_number,
        test_newtonschulz_preserves_direction_on_orthogonal_input,
        test_newtonschulz_sign_is_preserved,
        test_update_rms_is_consistent_across_shapes,
        test_rejects_non_matrix_params,
        test_weight_decay_shrinks_an_unused_param,
        test_split_params_routes_by_name,
        test_orthogonalization_beats_raw_momentum,
        test_insensitive_to_target_conditioning,
        test_optimizer_converges_and_state_persists,
    ]:
        print(f"ok  {fn.__name__}" if (fn() or True) else "")
    print("\nall checks passed")
