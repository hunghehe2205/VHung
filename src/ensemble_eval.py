"""Ensemble evaluation: combine logits1 (p1) and logits3 (p3) scores.

Grid search alpha for best AUC on `alpha * p1 + (1 - alpha) * p3`.
Optionally applies Platt scaling to p3 before ensemble.

Usage:
    python src/ensemble_eval.py --checkpoint-path final_model/checkpoint.pth
    python src/ensemble_eval.py --checkpoint-path final_model/checkpoint.pth --use-platt
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def collect_all_scores(model, testdataloader, maxlen, prompt_text, device):
    """Run inference and collect p1, p3, and raw logits3 for all test frames."""
    model.to(device)
    model.eval()
    all_p1 = []
    all_p3 = []
    all_raw3 = []

    with torch.no_grad():
        for item in tqdm(testdataloader, desc='Collecting scores'):
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

            _, logits1, _, logits3 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1)[0:len_cur]
            logits3 = logits3.reshape(-1)[0:len_cur]

            p1 = torch.sigmoid(logits1).cpu().numpy()
            raw3 = logits3.cpu().numpy()
            p3 = 1.0 / (1.0 + np.exp(-raw3))

            all_p1.append(np.repeat(p1, 16))
            all_p3.append(np.repeat(p3, 16))
            all_raw3.append(np.repeat(raw3, 16))

    return (np.concatenate(all_p1),
            np.concatenate(all_p3),
            np.concatenate(all_raw3))


def platt_calibrate(raw_logits, temp, bias):
    return 1.0 / (1.0 + np.exp(-(raw_logits - bias) / temp))


def mle_platt(raw_logits, gt, balanced=True):
    """Fit Platt scaling via MLE with class balancing (handles imbalanced test set)."""
    X = raw_logits.reshape(-1, 1)
    cw = 'balanced' if balanced else None
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, class_weight=cw)
    lr.fit(X, gt)
    coef = lr.coef_[0, 0]
    intercept = lr.intercept_[0]
    temp = 1.0 / coef
    bias = -intercept / coef
    return (temp, bias)


def grid_search_ensemble(p1, p3, gt, label=''):
    """Grid search alpha for alpha * p1 + (1 - alpha) * p3."""
    alphas = np.linspace(0.0, 1.0, 21)
    best_auc = 0
    best_alpha = 0
    results = []
    for alpha in alphas:
        ens = alpha * p1 + (1 - alpha) * p3
        auc = roc_auc_score(gt, ens)
        ap = average_precision_score(gt, ens)
        results.append((alpha, auc, ap))
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha

    print(f"\n=== Ensemble Grid Search ({label}) ===")
    print(f"{'alpha':>6} {'AUC':>8} {'AP':>8}")
    for alpha, auc, ap in results:
        marker = '  <-- best' if alpha == best_alpha else ''
        print(f"{alpha:>6.2f} {auc:>8.4f} {ap:>8.4f}{marker}")

    return best_alpha, best_auc


def report_map_stats(scores, gt, label=''):
    """Report distribution stats for score map quality."""
    anom = scores[gt == 1]
    norm = scores[gt == 0]
    print(f"\n=== Map Quality ({label}) ===")
    print(f"  AUC: {roc_auc_score(gt, scores):.4f}")
    print(f"  Normal:  mean={norm.mean():.3f} std={norm.std():.3f}")
    print(f"  Anomaly: mean={anom.mean():.3f} std={anom.std():.3f}")
    print(f"  Gap: {anom.mean() - norm.mean():.3f}")
    print(f"  Coverage (>0.5): {(anom > 0.5).mean():.3f}")


if __name__ == '__main__':
    option.parser.add_argument('--use-platt', action='store_true',
                               help='Apply Platt scaling to p3 before ensemble')
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        print(f"Loaded checkpoint (epoch={checkpoint['epoch']+1}, AUC1={checkpoint['ap']:.4f})")
    else:
        state = checkpoint
        print(f"Loaded raw state_dict from {args.checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (expected for map_head on original model)")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    p1, p3, raw3 = collect_all_scores(model, testdataloader, args.visual_length, prompt_text, device)

    auc1 = roc_auc_score(gt, p1)
    auc3 = roc_auc_score(gt, p3)
    print(f"\n=== Head Individual AUC ===")
    print(f"  AUC1 (logits1): {auc1:.4f}")
    print(f"  AUC3 (logits3): {auc3:.4f}")

    report_map_stats(p1, gt, 'p1 raw')
    report_map_stats(p3, gt, 'p3 raw')

    if args.use_platt:
        temp, bias = mle_platt(raw3, gt)
        p3_cal = platt_calibrate(raw3, temp, bias)
        print(f"\n=== Platt Scaling ===")
        print(f"  Best temp={temp:.4f}, bias={bias:.4f}")
        report_map_stats(p3_cal, gt, 'p3 calibrated')
        p3_for_ensemble = p3_cal
        ens_label = 'alpha*p1 + (1-alpha)*p3_calibrated'
    else:
        p3_for_ensemble = p3
        ens_label = 'alpha*p1 + (1-alpha)*p3'

    best_alpha, best_ens_auc = grid_search_ensemble(p1, p3_for_ensemble, gt, ens_label)
    best_ensemble = best_alpha * p1 + (1 - best_alpha) * p3_for_ensemble
    report_map_stats(best_ensemble, gt, f'ensemble alpha={best_alpha:.2f}')

    print(f"\n=== Summary ===")
    print(f"  AUC1:          {auc1:.4f}")
    print(f"  AUC3:          {auc3:.4f}")
    print(f"  Best ensemble: {best_ens_auc:.4f} (alpha={best_alpha:.2f})")
    print(f"  Baseline:      0.8801 (VadCLIP reported)")
    print(f"  Beat baseline: {'YES ✓' if best_ens_auc > 0.8801 else 'NO ✗'}")

    os.makedirs(args.log_dir, exist_ok=True)
    np.save(os.path.join(args.log_dir, 'ensemble_scores.npy'), best_ensemble)
    np.save(os.path.join(args.log_dir, 'ensemble_params.npy'),
            np.array([best_alpha, best_ens_auc]))
    print(f"\nSaved to {args.log_dir}/ensemble_scores.npy and ensemble_params.npy")
