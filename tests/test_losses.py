import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import (CLAS2, CLASM, get_lambdas, tcn_bce, tcn_ctr,  # noqa: E402
                   tcn_dice)


def test_tcn_bce_scalar_positive():
    logits = torch.randn(2, 256)
    y_soft = torch.rand(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    pw = torch.tensor([6.0])
    loss = tcn_bce(logits, y_soft, mask, pw)
    assert loss.dim() == 0 and loss.item() > 0


def test_tcn_dice_zero_for_all_normal_batch():
    probs = torch.rand(2, 256)
    y = torch.zeros(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    assert tcn_dice(probs, y, mask).item() == 0.0


def test_tcn_ctr_margin_violation_positive():
    probs = torch.zeros(1, 256)
    probs[0, 0:100] = 0.1
    probs[0, 100:] = 0.9
    y = torch.zeros(1, 256)
    y[0, 0:100] = 1.0
    mask = torch.ones(1, 256, dtype=torch.bool)
    loss = tcn_ctr(probs, y, mask, margin=0.3)
    assert loss.item() > 0.3


def test_get_lambdas_three_phases():
    assert get_lambdas(0, 3, 6) == (0.0, 0.0, 0.0, 1)
    assert get_lambdas(2, 3, 6) == (0.0, 0.0, 0.0, 1)
    assert get_lambdas(3, 3, 6) == (1.0, 0.0, 0.0, 2)
    assert get_lambdas(5, 3, 6) == (1.0, 0.0, 0.0, 2)
    assert get_lambdas(6, 3, 6) == (1.0, 1.0, 1.0, 3)
    assert get_lambdas(19, 3, 6) == (1.0, 1.0, 1.0, 3)
