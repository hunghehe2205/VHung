import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch

from dbranch import DBranch


def test_forward_returns_correct_shape():
    head = DBranch(in_dim=512)
    x = torch.randn(2, 64, 512)
    s = head(x)
    assert s.shape == (2, 64)


def test_forward_values_in_unit_interval():
    head = DBranch(in_dim=512)
    x = torch.randn(2, 64, 512)
    s = head(x)
    assert (s >= 0).all() and (s <= 1).all()


def test_padding_mask_zeroes_padded_positions():
    head = DBranch(in_dim=512)
    x = torch.randn(2, 8, 512)
    pad = torch.tensor([[False] * 5 + [True] * 3,
                        [False] * 6 + [True] * 2])
    s = head(x, padding_mask=pad)
    assert (s[0, 5:] == 0).all()
    assert (s[1, 6:] == 0).all()
    assert (s[0, :5] >= 0).all() and (s[0, :5] <= 1).all()


def test_gradient_flows_back_to_input():
    head = DBranch(in_dim=512)
    x = torch.randn(1, 16, 512, requires_grad=True)
    s = head(x)
    s.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_param_count_under_500k():
    head = DBranch(in_dim=512)
    n = sum(p.numel() for p in head.parameters())
    assert n < 500_000, f"D-Branch should be lightweight; got {n} params"
