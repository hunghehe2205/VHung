import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import soft_iou_loss, frame_bce_loss


def test_soft_iou_perfect_overlap_is_zero():
    probs = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(probs).bool()
    loss = soft_iou_loss(probs, target, mask)
    assert loss.item() < 0.05


def test_soft_iou_zero_overlap_is_high():
    probs = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    mask = torch.ones_like(probs).bool()
    loss = soft_iou_loss(probs, target, mask)
    assert loss.item() > 0.9


def test_soft_iou_respects_mask():
    probs = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask_full = torch.ones_like(probs).bool()
    mask_first_two = torch.tensor([[True, True, False, False]])
    l_full = soft_iou_loss(probs, target, mask_full).item()
    l_masked = soft_iou_loss(probs, target, mask_first_two).item()
    # Masking out the irrelevant half should reduce the loss.
    assert l_masked < l_full


def test_frame_bce_uses_pos_weight():
    logits = torch.tensor([[-2.0, 2.0, -2.0, 2.0]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])          # all predictions wrong
    mask = torch.ones_like(logits).bool()
    pw_low = torch.tensor([1.0])
    pw_high = torch.tensor([10.0])
    l_low = frame_bce_loss(logits, y, mask, pw_low).item()
    l_high = frame_bce_loss(logits, y, mask, pw_high).item()
    assert l_high > l_low


from train import get_lambda


def test_get_lambda_phases():
    assert get_lambda(0, 3, 6, 0.1, 0.1) == (0.0, 0.0)
    assert get_lambda(2, 3, 6, 0.1, 0.1) == (0.0, 0.0)
    assert get_lambda(3, 3, 6, 0.1, 0.1) == (0.1, 0.0)
    assert get_lambda(5, 3, 6, 0.1, 0.1) == (0.1, 0.0)
    assert get_lambda(6, 3, 6, 0.1, 0.1) == (0.1, 0.1)
    assert get_lambda(9, 3, 6, 0.1, 0.1) == (0.1, 0.1)
