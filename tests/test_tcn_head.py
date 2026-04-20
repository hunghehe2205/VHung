import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import CLIPVAD  # noqa: E402


def _make_model():
    return CLIPVAD(num_class=14, embed_dim=512, visual_length=256,
                   visual_width=512, visual_head=1, visual_layers=2,
                   attn_window=8, prompt_prefix=10, prompt_postfix=10,
                   device='cpu')


def test_forward_returns_4_tensors_including_tcn_logits():
    model = _make_model().eval()
    visual = torch.randn(2, 256, 512)
    lengths = torch.tensor([128, 200])
    text = ['normal', 'abuse', 'arrest', 'arson', 'assault', 'burglary',
            'explosion', 'fighting', 'roadAccidents', 'robbery', 'shooting',
            'shoplifting', 'stealing', 'vandalism']
    with torch.no_grad():
        out = model(visual, None, text, lengths)
    assert len(out) == 4, f'forward must return 4 tensors, got {len(out)}'
    _, logits1, logits2, tcn_logits = out
    assert logits1.shape == (2, 256, 1)
    assert logits2.shape == (2, 256, 14)
    assert tcn_logits.shape == (2, 256, 1), \
        f'tcn_logits must be [B,T,1], got {tuple(tcn_logits.shape)}'


def test_tcn_head_parameters_exist():
    model = _make_model()
    names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith('tcn.') for n in names), \
        'TCN head params not registered'
    assert not any('start_head' in n or 'end_head' in n for n in names), \
        'start_head / end_head must be removed'
