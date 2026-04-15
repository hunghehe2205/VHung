"""Dense per-frame losses for D-Branch.

Score tensor convention throughout: ``s`` has shape ``(B, T)`` with values in
``[0, 1]``; ``y`` has the same shape as binary frame labels in ``{0, 1}``;
``lengths`` is a 1-D LongTensor of valid lengths per video. Padded positions
(indices ``>= lengths[i]``) are excluded from every term.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


def _iter_video_slices(s: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor):
    for i in range(s.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        yield i, L, s[i, :L], y[i, :L]


def bce_loss(s: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Per-frame binary cross-entropy on valid positions, averaged over frames.

    Auto pos_weight = (#neg / #pos) computed per batch handles the heavy
    normal-frame imbalance without an external hyperparameter.
    """
    parts = []
    counts = []
    for _, L, s_i, y_i in _iter_video_slices(s, y, lengths):
        parts.append((s_i, y_i))
        counts.append(L)
    if not parts:
        return s.new_zeros((), requires_grad=True)

    s_flat = torch.cat([p[0] for p in parts])
    y_flat = torch.cat([p[1] for p in parts])
    n_pos = y_flat.sum().clamp(min=1.0)
    n_neg = (1.0 - y_flat).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).detach()
    # binary_cross_entropy expects sigmoid output (already in [0,1]).
    weight = torch.where(y_flat > 0.5, pos_weight, torch.ones_like(y_flat))
    s_safe = s_flat.clamp(EPS, 1.0 - EPS)
    bce = -(y_flat * torch.log(s_safe) + (1.0 - y_flat) * torch.log(1.0 - s_safe))
    return (weight * bce).mean()


def _soft_min(x: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable approximation of min via -logsumexp(-x / T) * T."""
    return -torch.logsumexp(-x * temperature, dim=0) / temperature


def _soft_max(x: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.logsumexp(x * temperature, dim=0) / temperature


def margin_loss(
    s: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    m: float = 0.3,
    temperature: float = 5.0,
) -> torch.Tensor:
    """Per-video valley margin: ``max(0, m - (soft_min s_in - soft_max s_out))``.

    Skips videos with no event frames (normal videos) or no normal frames
    (fully anomalous, very rare).

    ``temperature`` controls how sharp the soft-min/max is. T=5 keeps gradient
    spread over many in-event frames; T=10+ collapses gradient onto the 1-2
    extreme frames per side, which is harmful for long events (50+ clips).
    """
    losses = []
    for _, L, s_i, y_i in _iter_video_slices(s, y, lengths):
        in_event = y_i > 0.5
        out_event = ~in_event
        if not in_event.any() or not out_event.any():
            continue
        s_in = s_i[in_event]
        s_out = s_i[out_event]
        gap = _soft_min(s_in, temperature) - _soft_max(s_out, temperature)
        losses.append(F.relu(m - gap))
    if not losses:
        return s.new_zeros((), requires_grad=True)
    return torch.stack(losses).mean()


def soft_dice_loss(
    s: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Soft Dice: ``1 - 2·Σ(s·y) / (Σs + Σy + ε)`` per video, averaged.

    Skips videos with no event frames (normal): Dice is undefined when the
    target is all-zero (would always be 0 regardless of ``s``).
    """
    losses = []
    for _, L, s_i, y_i in _iter_video_slices(s, y, lengths):
        if y_i.sum() < 0.5:
            continue
        inter = (s_i * y_i).sum()
        denom = s_i.sum() + y_i.sum() + EPS
        dice = (2.0 * inter) / denom
        losses.append(1.0 - dice)
    if not losses:
        return s.new_zeros((), requires_grad=True)
    return torch.stack(losses).mean()


def variance_loss(
    s: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Plateau penalty: per-event variance of in-event scores, averaged.

    Encourages each event region to be approximately flat (uniform plateau).
    A single contiguous run of ``y == 1`` is treated as one event; multiple
    runs in the same video are penalised independently then averaged.

    Implementation is pure-tensor (no host transfer) for GPU/AMP safety:
    span boundaries are derived via ``torch.diff`` on the padded indicator,
    and start/end indices are extracted with ``nonzero`` on-device.
    """
    per_event_vars = []
    for _, L, s_i, y_i in _iter_video_slices(s, y, lengths):
        indicator = (y_i > 0.5).to(s_i.dtype)
        zero = s_i.new_zeros(1)
        padded = torch.cat([zero, indicator, zero])
        diff = padded[1:] - padded[:-1]
        starts = (diff > 0.5).nonzero(as_tuple=False).flatten()
        ends = (diff < -0.5).nonzero(as_tuple=False).flatten()
        for a, b in zip(starts.tolist(), ends.tolist()):
            if b - a < 2:
                continue
            per_event_vars.append(s_i[a:b].var(unbiased=False))
    if not per_event_vars:
        return s.new_zeros((), requires_grad=True)
    return torch.stack(per_event_vars).mean()


def dense_loss_total(
    s: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    w_bce: float = 1.0,
    w_margin: float = 0.5,
    w_dice: float = 0.5,
    w_var: float = 0.1,
    margin_m: float = 0.3,
    margin_temp: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Aggregate dense loss = weighted sum of BCE + Margin + Dice + Var.

    Returns ``(total_loss_tensor, per_term_scalar_dict)``. Per-term values are
    detached floats for logging; only the total tensor carries gradient.
    """
    l_bce = bce_loss(s, y, lengths)
    l_margin = margin_loss(s, y, lengths, m=margin_m, temperature=margin_temp)
    l_dice = soft_dice_loss(s, y, lengths)
    l_var = variance_loss(s, y, lengths)

    total = w_bce * l_bce + w_margin * l_margin + w_dice * l_dice + w_var * l_var
    parts = {
        'bce': float(l_bce.item()),
        'margin': float(l_margin.item()),
        'dice': float(l_dice.item()),
        'var': float(l_var.item()),
    }
    return total, parts
