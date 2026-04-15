import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import pytest
import torch

from losses_dbranch import (
    bce_loss,
    margin_loss,
    soft_dice_loss,
    variance_loss,
    dense_loss_total,
    focal_bce_loss,
    tversky_loss,
    hinge_coverage_loss,
)


# ---------- BCE ----------

def test_bce_zero_when_perfect():
    s = torch.tensor([[0.999, 0.999, 0.001, 0.001]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = bce_loss(s, y, lengths)
    assert loss.item() < 1e-2


def test_bce_high_when_inverted():
    s = torch.tensor([[0.001, 0.001, 0.999, 0.999]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = bce_loss(s, y, lengths)
    assert loss.item() > 3.0


def test_bce_ignores_padded_positions():
    # Length 2, but 4 positions — last 2 are pad and should not pollute loss.
    s = torch.tensor([[0.999, 0.001, 0.5, 0.5]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    lengths = torch.tensor([2])
    loss = bce_loss(s, y, lengths)
    assert loss.item() < 1e-2


def test_bce_gradient_flows():
    s = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    y = torch.tensor([[1.0, 1.0, 0.0]])
    lengths = torch.tensor([3])
    loss = bce_loss(s, y, lengths)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Margin ----------

def test_margin_zero_when_gap_exceeds_threshold():
    # in-event scores ~0.9, out-event scores ~0.1 → gap ~0.8 > m=0.3
    s = torch.tensor([[0.1, 0.1, 0.9, 0.9, 0.9, 0.1]])
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = margin_loss(s, y, lengths, m=0.3, temperature=10.0)
    assert loss.item() < 1e-2


def test_margin_positive_when_no_gap():
    # All scores ~0.5 → true gap = 0 → loss ≥ m. With soft-min/max at T=10
    # the approximation adds log(N_in)/T + log(N_out)/T extra penalty, so
    # the realised loss exceeds m. Behaviour we care about: strictly positive.
    s = torch.full((1, 6), 0.5)
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = margin_loss(s, y, lengths, m=0.3, temperature=10.0)
    assert loss.item() > 0.3


def test_margin_skips_normal_videos():
    # No event frames (y all zero) → margin undefined → skip → loss 0
    s = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = margin_loss(s, y, lengths, m=0.3, temperature=10.0)
    assert loss.item() == 0.0


def test_margin_gradient_flows():
    s = torch.tensor([[0.4, 0.4, 0.6, 0.6, 0.4]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([5])
    loss = margin_loss(s, y, lengths, m=0.3, temperature=10.0)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Soft Dice ----------

def test_dice_zero_when_perfect_overlap():
    s = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = soft_dice_loss(s, y, lengths)
    assert loss.item() < 0.05


def test_dice_high_when_disjoint():
    s = torch.tensor([[0.01, 0.01, 0.99, 0.99]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = soft_dice_loss(s, y, lengths)
    assert loss.item() > 0.9


def test_dice_gradient_flows():
    s = torch.tensor([[0.3, 0.4, 0.5, 0.6]], requires_grad=True)
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = soft_dice_loss(s, y, lengths)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Variance (plateau) ----------

def test_variance_zero_when_uniform_in_event():
    s = torch.tensor([[0.1, 0.8, 0.8, 0.8, 0.8, 0.1]])
    y = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = variance_loss(s, y, lengths)
    assert loss.item() < 1e-4


def test_variance_high_when_peaky_in_event():
    s = torch.tensor([[0.1, 0.99, 0.01, 0.99, 0.01, 0.1]])
    y = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = variance_loss(s, y, lengths)
    assert loss.item() > 0.1


def test_variance_skips_normal_videos():
    s = torch.tensor([[0.1, 0.5, 0.9, 0.3]])
    y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = variance_loss(s, y, lengths)
    assert loss.item() == 0.0


# ---------- Combined ----------

def test_dense_loss_total_returns_per_term_dict():
    s = torch.tensor([[0.4, 0.4, 0.6, 0.6, 0.4]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([5])
    total, parts = dense_loss_total(
        s, y, lengths,
        w_bce=1.0, w_margin=0.5, w_dice=0.5, w_var=0.1,
        margin_m=0.3, margin_temp=10.0,
    )
    for k in ('bce', 'margin', 'dice', 'var'):
        assert k in parts
        assert isinstance(parts[k], float)
    total.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Focal BCE ----------

def test_focal_bce_zero_when_perfect():
    s = torch.tensor([[0.999, 0.999, 0.001, 0.001]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = focal_bce_loss(s, y, lengths, gamma=2.0)
    assert loss.item() < 1e-3


def test_focal_bce_downweights_easy_frames():
    """Focal vs plain BCE: on a mostly-correct prediction, Focal loss should
    be strictly smaller because easy frames are down-weighted."""
    s = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    l_focal = focal_bce_loss(s, y, lengths, gamma=2.0).item()
    l_bce = bce_loss(s, y, lengths).item()
    assert l_focal < l_bce


def test_focal_bce_high_when_inverted():
    s = torch.tensor([[0.001, 0.001, 0.999, 0.999]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = focal_bce_loss(s, y, lengths, gamma=2.0)
    assert loss.item() > 1.0


def test_focal_bce_gamma_zero_equals_bce():
    """γ=0 removes the focal modulating factor, reducing to plain BCE."""
    s = torch.tensor([[0.7, 0.3, 0.6, 0.2]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    lengths = torch.tensor([4])
    l_focal = focal_bce_loss(s, y, lengths, gamma=0.0).item()
    l_bce = bce_loss(s, y, lengths).item()
    assert abs(l_focal - l_bce) < 1e-4


def test_focal_bce_gradient_flows():
    s = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    y = torch.tensor([[1.0, 1.0, 0.0]])
    lengths = torch.tensor([3])
    loss = focal_bce_loss(s, y, lengths, gamma=2.0)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Tversky ----------

def test_tversky_zero_when_perfect():
    s = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = tversky_loss(s, y, lengths, alpha=0.3, beta=0.7)
    assert loss.item() < 0.05


def test_tversky_penalises_fn_more_when_beta_greater_than_alpha():
    """Missing one event frame (FN) should cost more than one false alarm (FP)
    when β > α."""
    lengths = torch.tensor([4])

    # Case A: one false alarm (FP) — extra coverage outside event
    s_fp = torch.tensor([[0.0, 1.0, 1.0, 0.9]])
    y    = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    # Case B: one missed event (FN) — one event frame with low score
    s_fn = torch.tensor([[0.0, 1.0, 0.1, 0.0]])

    l_fp = tversky_loss(s_fp, y, lengths, alpha=0.3, beta=0.7).item()
    l_fn = tversky_loss(s_fn, y, lengths, alpha=0.3, beta=0.7).item()

    assert l_fn > l_fp


def test_tversky_alpha_equals_beta_gives_dice():
    """Tversky(α=0.5, β=0.5) reduces to Dice."""
    s = torch.tensor([[0.7, 0.8, 0.2, 0.3]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    l_tversky = tversky_loss(s, y, lengths, alpha=0.5, beta=0.5).item()
    l_dice = soft_dice_loss(s, y, lengths).item()
    assert abs(l_tversky - l_dice) < 1e-4


def test_tversky_skips_normal_videos():
    s = torch.tensor([[0.1, 0.2, 0.1, 0.1]])
    y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = tversky_loss(s, y, lengths, alpha=0.3, beta=0.7)
    assert loss.item() == 0.0


def test_tversky_gradient_flows():
    s = torch.tensor([[0.3, 0.4, 0.5, 0.6]], requires_grad=True)
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = tversky_loss(s, y, lengths, alpha=0.3, beta=0.7)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


# ---------- Hinge coverage ----------

def test_hinge_zero_when_all_on_correct_side():
    """All event frames ≥ 0.5, all non-event frames ≤ 0.5 → no violation."""
    s = torch.tensor([[0.2, 0.3, 0.8, 0.9, 0.7, 0.1]])
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    assert loss.item() < 1e-4


def test_hinge_positive_on_coverage_violation():
    """Event frame below threshold → positive loss."""
    s = torch.tensor([[0.2, 0.3, 0.2, 0.9, 0.7, 0.1]])  # one event frame is 0.2 < 0.5
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    assert loss.item() > 1e-3


def test_hinge_positive_on_exclusion_violation():
    """Non-event frame above threshold → positive loss."""
    s = torch.tensor([[0.2, 0.9, 0.8, 0.9, 0.7, 0.1]])  # one non-event frame is 0.9 > 0.5
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([6])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    assert loss.item() > 1e-3


def test_hinge_gradient_only_on_violating_frames():
    """Frames already on the correct side should receive zero gradient."""
    s = torch.tensor([[0.1, 0.9, 0.2, 0.8]], requires_grad=True)
    # y=1 for indices 1, 3 (both already high), y=0 for 0, 2 (both already low)
    y = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    lengths = torch.tensor([4])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    # All correct → loss zero, gradient zero
    assert loss.item() < 1e-4
    loss.backward()
    assert s.grad.abs().sum().item() < 1e-4


def test_hinge_gradient_flows_on_violations():
    s = torch.tensor([[0.3, 0.3, 0.7, 0.7]], requires_grad=True)
    # y=1 at 0,2 — but s=0.3 at index 0 is below threshold (coverage violation)
    # y=0 at 1,3 — but s=0.7 at index 3 is above threshold (exclusion violation)
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    lengths = torch.tensor([4])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0


def test_hinge_handles_normal_video():
    """Normal video: only exclusion term applies."""
    s = torch.tensor([[0.1, 0.2, 0.15, 0.6]])  # last frame violates
    y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    lengths = torch.tensor([4])
    loss = hinge_coverage_loss(s, y, lengths, threshold=0.5)
    assert loss.item() > 1e-3


# ---------- dense_loss_total extended ----------

def test_dense_loss_total_includes_new_losses():
    s = torch.tensor([[0.4, 0.4, 0.6, 0.6, 0.4]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    lengths = torch.tensor([5])
    total, parts = dense_loss_total(
        s, y, lengths,
        w_bce=0.0, w_margin=0.0, w_dice=0.0, w_var=0.0,
        w_focal=1.0, focal_gamma=2.0,
        w_tversky=1.0, tversky_alpha=0.3, tversky_beta=0.7,
        w_hinge=1.0, hinge_threshold=0.5,
    )
    for k in ('focal', 'tversky', 'hinge'):
        assert k in parts
        assert isinstance(parts[k], float)
    total.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0
