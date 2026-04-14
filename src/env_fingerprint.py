"""Env numerics diagnostic for VadCLIP.

Runs inference under 3 configs (default / cudnn deterministic / math SDP)
against `model_ucf.pth`, dumps layer activations for a fixed input, and
compares AUC + activation hashes. Purpose: pinpoint the 0.8736 vs 0.8801 gap.

Usage:
    python src/env_fingerprint.py --model-path final_model/model_ucf.pth
"""
import os
import hashlib
import contextlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def hash_tensor(t):
    arr = t.detach().cpu().to(torch.float64).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def summarize_tensor(t):
    x = t.detach().cpu().to(torch.float64)
    return {
        'shape': tuple(x.shape),
        'mean': float(x.mean()),
        'std': float(x.std()),
        'min': float(x.min()),
        'max': float(x.max()),
        'abs_max': float(x.abs().max()),
        'hash': hash_tensor(t),
    }


@contextlib.contextmanager
def sdp_backend(enable_flash, enable_math, enable_mem_efficient):
    """Force attention SDP kernel backend (torch 2.0+ only)."""
    sdp = getattr(torch.backends.cuda, 'sdp_kernel', None)
    if sdp is None:
        yield
        return
    try:
        with sdp(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient,
        ):
            yield
    except TypeError:
        # Older signature or unavailable kwargs → skip
        yield


def set_config(name):
    """Apply a named env config and return sdp context."""
    # Reset TF32 defaults
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    if name == 'default':
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return sdp_backend(True, True, True)
    elif name == 'cudnn_det':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        return sdp_backend(True, True, True)
    elif name == 'no_tf32':
        # Disable TF32 (Ampere+ GPUs default to TF32 for matmul/cudnn);
        # original paper likely trained on older GPU without TF32 → FP32 numerics differ.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        return sdp_backend(False, True, False)
    elif name == 'math_sdp':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        return sdp_backend(False, True, False)
    else:
        raise ValueError(name)


def register_hooks(model):
    """Attach forward hooks on major layers; returns dict and a remover."""
    captured = {}
    handles = []

    def make_hook(name):
        def hook(_m, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if torch.is_tensor(out):
                captured[name] = out.detach()
        return hook

    targets = {
        'temporal': model.temporal,
        'gc1': model.gc1,
        'gc2': model.gc2,
        'gc3': model.gc3,
        'gc4': model.gc4,
        'linear': model.linear,
        'mlp1': model.mlp1,
        'mlp2': model.mlp2,
        'classifier': model.classifier,
        'map_head': model.map_head,
    }
    for name, module in targets.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    def remove():
        for h in handles:
            h.remove()

    return captured, remove


def run_fixed_sample(model, testdataloader, maxlen, prompt_text, device, sample_idx=0):
    """Forward the `sample_idx`-th test video; return captured activations + logits1."""
    model.eval()
    captured, remove = register_hooks(model)
    try:
        with torch.no_grad():
            for i, item in enumerate(testdataloader):
                if i != sample_idx:
                    continue
                visual = item[0].squeeze(0)
                length = int(item[2])
                len_cur = length
                if len_cur < maxlen:
                    visual = visual.unsqueeze(0)
                visual = visual.to(device)

                lengths = torch.zeros(int(length / maxlen) + 1)
                for j in range(int(length / maxlen) + 1):
                    if j == 0 and length < maxlen:
                        lengths[j] = length
                    elif j == 0 and length > maxlen:
                        lengths[j] = maxlen
                        length -= maxlen
                    elif length > maxlen:
                        lengths[j] = maxlen
                        length -= maxlen
                    else:
                        lengths[j] = length
                lengths = lengths.to(int)
                padding_mask = get_batch_mask(lengths, maxlen).to(device)

                _, logits1, _, _ = model(visual, padding_mask, prompt_text, lengths)
                logits1 = logits1.reshape(-1)[0:len_cur]
                break
    finally:
        remove()

    summaries = {name: summarize_tensor(t) for name, t in captured.items()}
    summaries['__logits1__'] = summarize_tensor(logits1)
    return summaries


def eval_full_auc(model, testdataloader, maxlen, prompt_text, gt, device):
    model.eval()
    all_p1 = []
    with torch.no_grad():
        for item in tqdm(testdataloader, desc='AUC eval'):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, _, _ = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1)[0:len_cur]
            prob1 = torch.sigmoid(logits1).cpu().numpy()
            all_p1.append(np.repeat(prob1, 16))

    p1_flat = np.concatenate(all_p1)
    return float(roc_auc_score(gt, p1_flat)), p1_flat


def diff_summaries(base, other, name):
    print(f"\n--- activation diff: default vs {name} ---")
    print(f"  {'layer':>12} {'mean_diff':>12} {'max_diff':>12} {'hash_match':>10}")
    for layer in base:
        b = base[layer]
        o = other[layer]
        mean_diff = abs(b['mean'] - o['mean'])
        max_diff = abs(b['max'] - o['max'])
        same = (b['hash'] == o['hash'])
        print(f"  {layer:>12} {mean_diff:>12.2e} {max_diff:>12.2e} {str(same):>10}")


if __name__ == '__main__':
    option.parser.add_argument('--fixed-sample-idx', default=0, type=int)
    option.parser.add_argument('--configs', default='default,cudnn_det,no_tf32',
                               help='Comma-separated configs to test '
                                    '(default, cudnn_det, no_tf32, math_sdp)')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)

    config_names = args.configs.split(',')
    all_summaries = {}
    all_aucs = {}

    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cudnn: {torch.backends.cudnn.version()}")
        print(f"device: {torch.cuda.get_device_name(0)}")

    for cfg in config_names:
        print(f"\n========== Config: {cfg} ==========")
        ctx = set_config(cfg)

        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                        args.visual_head, args.visual_layers, args.attn_window,
                        args.prompt_prefix, args.prompt_postfix, device)
        ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.to(device)

        with ctx:
            summaries = run_fixed_sample(model, testdataloader, args.visual_length,
                                         prompt_text, device, args.fixed_sample_idx)
            auc, _ = eval_full_auc(model, testdataloader, args.visual_length,
                                   prompt_text, gt, device)

        all_summaries[cfg] = summaries
        all_aucs[cfg] = auc

        print(f"\n  Activations on sample {args.fixed_sample_idx}:")
        for layer, s in summaries.items():
            print(f"    {layer:>12} shape={str(s['shape']):>20} "
                  f"mean={s['mean']:>+.4e} std={s['std']:.4e} "
                  f"absmax={s['abs_max']:.4e} hash={s['hash']}")
        print(f"  Full-test AUC1: {auc:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n\n========== Summary ==========")
    print(f"  {'config':>14} {'AUC':>8} {'delta vs default':>18}")
    base_auc = all_aucs[config_names[0]]
    for cfg in config_names:
        delta = all_aucs[cfg] - base_auc
        print(f"  {cfg:>14} {all_aucs[cfg]:>8.4f} {delta:>+18.4f}")
    print(f"  paper target: 0.8801")

    base = all_summaries[config_names[0]]
    for cfg in config_names[1:]:
        diff_summaries(base, all_summaries[cfg], cfg)
