import torch
import pytest

# sys.path needed because train.py uses `from model import CLIPVAD` etc. (src/ on path)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import map_mass_ratio_loss, map_density_loss


def _make(logits_vals, mask_vals):
    logits = torch.tensor(logits_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    mask = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(0)  # (1, T)
    lengths = torch.tensor([len(logits_vals)])
    return logits, mask, lengths


def test_mass_ratio_loss_zero_when_mass_concentrated():
    # All mass inside event, margin=0.1 → EMR=1.0 ≥ ETR+0.1 ⇒ loss = 0
    T = 20
    logits = [-10.0] * T
    for i in range(5, 15):
        logits[i] = 10.0  # sigmoid ≈ 1.0
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_mass_ratio_loss(L, M, Lens, margin=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_mass_ratio_loss_positive_when_uniform():
    # Uniform mass → EMR ≈ ETR = 0.5 → loss = margin = 0.3
    T = 20
    logits = [0.0] * T  # sigmoid=0.5
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_mass_ratio_loss(L, M, Lens, margin=0.3)
    assert 0.25 < loss.item() < 0.35


def test_density_loss_zero_on_uniform():
    # Uniform inside event → normalized entropy = 1 → loss = 0
    T = 20
    logits = [0.0] * T  # sigmoid=0.5 uniformly
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_density_loss(L, M, Lens)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_density_loss_high_on_spike():
    # One frame high, rest zero inside event → entropy → 0 → loss → 1
    T = 20
    logits = [-10.0] * T
    logits[10] = 10.0
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_density_loss(L, M, Lens)
    assert loss.item() > 0.9
