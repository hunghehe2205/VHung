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
