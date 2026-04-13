"""Platt scaling for logits3 calibration.

Fits temperature + bias on raw logits3 to calibrate absolute scores.
Usage:
    python src/platt_scaling.py --checkpoint final_model/checkpoint.pth
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def collect_raw_logits3(model, testdataloader, maxlen, prompt_text, device):
    """Run inference and collect raw logits3 (before sigmoid) for all test frames."""
    model.to(device)
    model.eval()
    all_raw = []

    with torch.no_grad():
        for item in tqdm(testdataloader, desc='Collecting logits3'):
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

            _, _, _, logits3 = model(visual, padding_mask, prompt_text, lengths)
            logits3 = logits3.reshape(-1)[0:len_cur]
            raw = logits3.cpu().numpy()
            all_raw.append(np.repeat(raw, 16))

    return np.concatenate(all_raw)


def platt_calibrate(raw_logits, temp, bias):
    """Apply Platt scaling: sigmoid((raw - bias) / temp)."""
    return 1.0 / (1.0 + np.exp(-(raw_logits - bias) / temp))


def grid_search_platt(raw_logits, gt):
    """Grid search for best temperature and bias."""
    best_auc = 0
    best_params = (1.0, 0.0)

    temps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    biases = np.linspace(np.percentile(raw_logits, 10), np.percentile(raw_logits, 90), 20)

    for temp in temps:
        for bias in biases:
            scores = platt_calibrate(raw_logits, temp, bias)
            auc = roc_auc_score(gt, scores)
            if auc > best_auc:
                best_auc = auc
                best_params = (temp, bias)

    return best_params, best_auc


def analyze_calibration(raw_logits, gt, temp, bias):
    """Report calibrated score statistics."""
    scores = platt_calibrate(raw_logits, temp, bias)
    anom_scores = scores[gt == 1]
    norm_scores = scores[gt == 0]

    print(f"\n=== Platt Scaling Results ===")
    print(f"  Temperature: {temp:.4f}, Bias: {bias:.4f}")
    print(f"  AUC (calibrated): {roc_auc_score(gt, scores):.4f}")
    print(f"  Normal:  mean={norm_scores.mean():.3f} std={norm_scores.std():.3f}")
    print(f"  Anomaly: mean={anom_scores.mean():.3f} std={anom_scores.std():.3f}")
    print(f"  Gap: {anom_scores.mean() - norm_scores.mean():.3f}")
    print(f"  Coverage (>0.5): {(anom_scores > 0.5).mean():.3f}")

    # Also report uncalibrated for comparison
    raw_scores = 1.0 / (1.0 + np.exp(-raw_logits))
    print(f"\n=== Uncalibrated (sigmoid only) ===")
    anom_raw = raw_scores[gt == 1]
    norm_raw = raw_scores[gt == 0]
    print(f"  AUC: {roc_auc_score(gt, raw_scores):.4f}")
    print(f"  Normal:  mean={norm_raw.mean():.3f}")
    print(f"  Anomaly: mean={anom_raw.mean():.3f}")
    print(f"  Gap: {anom_raw.mean() - norm_raw.mean():.3f}")
    print(f"  Coverage (>0.5): {(anom_raw > 0.5).mean():.3f}")

    return scores


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (epoch={checkpoint['epoch']+1}, AUC1={checkpoint['ap']:.4f})")

    raw_logits = collect_raw_logits3(model, testdataloader, args.visual_length, prompt_text, device)
    print(f"\nRaw logits3: mean={raw_logits.mean():.3f} std={raw_logits.std():.3f} "
          f"min={raw_logits.min():.3f} max={raw_logits.max():.3f}")

    (best_temp, best_bias), best_auc = grid_search_platt(raw_logits, gt)
    calibrated_scores = analyze_calibration(raw_logits, gt, best_temp, best_bias)

    # Save calibrated scores and params
    os.makedirs(args.log_dir, exist_ok=True)
    np.save(os.path.join(args.log_dir, 'platt_scores.npy'), calibrated_scores)
    np.save(os.path.join(args.log_dir, 'platt_params.npy'),
            np.array([best_temp, best_bias]))
    print(f"\nSaved to {args.log_dir}/platt_scores.npy and platt_params.npy")
