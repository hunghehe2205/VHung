import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from tqdm import tqdm

from model import CLIPVAD
from test import test, LABEL_MAP
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import option


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def soft_iou_loss(probs, target, mask, eps=1e-6):
    """Sequence-level soft temporal IoU loss.
    probs: [B, T] in [0,1] — sigmoid(logits1).
    target: [B, T] in {0,1} — y_bin.
    mask: [B, T] bool — True for valid frames.
    Returns scalar mean across batch.
    """
    mask_f = mask.float()
    probs = probs * mask_f
    target = target * mask_f
    inter = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def tv_smoothness_loss(probs, mask):
    """Total-variation temporal smoothness on [B, T] probs.
    Penalizes |p_t - p_{t-1}| over frames where both t and t-1 are valid.
    Returns scalar = mean per-transition absolute difference.
    """
    diffs = (probs[:, 1:] - probs[:, :-1]).abs()
    valid = (mask[:, 1:] & mask[:, :-1]).float()
    return (diffs * valid).sum() / valid.sum().clamp_min(1.0)


def dice_loss_anomaly(probs, target, mask, eps=1.0):
    """Soft Dice on [B, T] probs, reduced over anomaly videos only.
    Normal videos (all-zero target) are skipped to avoid redundant signal
    with BCE. eps=1 for smoothing and non-zero gradient when union small.
    """
    mask_f = mask.float()
    p = probs * mask_f
    y = target * mask_f
    inter = (p * y).sum(-1)
    denom = p.sum(-1) + y.sum(-1)
    dice = (2.0 * inter + eps) / (denom + eps)
    has_pos = (y.sum(-1) > 0).float()
    return ((1.0 - dice) * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def within_video_contrast_loss(probs, target, mask, margin=0.3):
    """Per anomaly video: mean prob inside GT must exceed mean prob outside
    by at least `margin`. Forces peak AT the correct location rather than
    uniformly-high output. Normal videos skipped (no inside region).
    """
    valid = mask.float()
    inside = (target > 0.5).float() * valid
    outside = (1.0 - (target > 0.5).float()) * valid
    has_inside = (inside.sum(-1) > 0).float()
    inside_mean = (probs * inside).sum(-1) / inside.sum(-1).clamp_min(1.0)
    outside_mean = (probs * outside).sum(-1) / outside.sum(-1).clamp_min(1.0)
    gap_loss = F.relu(margin - (inside_mean - outside_mean))
    return (gap_loss * has_inside).sum() / has_inside.sum().clamp_min(1.0)


def frame_bce_loss(logits, target, mask, pos_weight):
    """Frame-level binary BCE on logits1 with scalar pos_weight.
    logits: [B, T] raw logits.
    target: [B, T] {0,1}.
    mask: [B, T] bool.
    pos_weight: 1-D tensor [1].
    """
    logits_m = logits[mask]
    target_m = target[mask]
    return F.binary_cross_entropy_with_logits(
        logits_m, target_m, pos_weight=pos_weight.to(logits_m.device))


def focal_bce_loss(logits, target, mask, pos_weight, gamma=2.0):
    """Focal BCE on logits1. Combines pos_weight (class balance) with
    focal modulation (1 - p_t)^gamma to emphasize hard frames.
    """
    logits_m = logits[mask]
    target_m = target[mask]
    bce = F.binary_cross_entropy_with_logits(
        logits_m, target_m, pos_weight=pos_weight.to(logits_m.device),
        reduction='none')
    p = torch.sigmoid(logits_m)
    p_t = p * target_m + (1.0 - p) * (1.0 - target_m)
    focal_w = (1.0 - p_t).clamp_min(1e-6) ** gamma
    return (focal_w * bce).mean()


def get_lambda(epoch, phase1_epochs, phase2_epochs, lambda1, lambda2):
    """2-phase schedule for extra losses.
    epoch < phase1_epochs: return (0, 0)                # MIL-only warmup
    epoch >= phase1_epochs: return (lambda1, lambda2)    # BCE + BSN + Dice/Contrast all active
    phase2_epochs kept in signature for CLI compat but unused.
    """
    if epoch < phase1_epochs:
        return 0.0, 0.0
    else:
        return float(lambda1), float(lambda2)


def build_boundary_targets(y_bin, mask, sigma=1.0):
    """Derive start/end targets from y_bin transitions, Gaussian-smoothed.
    0->1 = start, 1->0 = end.  Returns (start_tgt, end_tgt) each [B, T].
    """
    B, T = y_bin.shape
    device = y_bin.device
    y_pad = F.pad(y_bin, (1, 0), value=0.0)       # [B, T+1]
    diff = y_pad[:, 1:] - y_pad[:, :-1]           # [B, T]
    start_hard = (diff > 0.5).float()              # 0->1 transition
    # end: 1->0 transition (diff < -0.5) shifted left by 1 so it lands
    # on the LAST positive frame rather than the first negative frame.
    end_diff = (diff < -0.5).float()
    end_hard = torch.zeros_like(end_diff)
    end_hard[:, :-1] = end_diff[:, 1:]
    # Gaussian smooth
    k = int(2 * np.ceil(3 * sigma) + 1)
    coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kernel = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    kernel = kernel / kernel.max()
    kernel = kernel.view(1, 1, k)
    start_tgt = F.conv1d(start_hard.unsqueeze(1), kernel,
                         padding=k // 2).squeeze(1).clamp(0, 1) * mask.float()
    end_tgt = F.conv1d(end_hard.unsqueeze(1), kernel,
                       padding=k // 2).squeeze(1).clamp(0, 1) * mask.float()
    return start_tgt, end_tgt


def boundary_bce_loss(start_logits, end_logits, y_bin, mask, sigma=1.0,
                      pos_weight=10.0):
    """BCE for start/end heads. Only computed on valid (masked) frames."""
    start_tgt, end_tgt = build_boundary_targets(y_bin, mask, sigma=sigma)
    pw = torch.tensor([pos_weight], device=y_bin.device)
    s_loss = F.binary_cross_entropy_with_logits(
        start_logits.squeeze(-1)[mask], start_tgt[mask], pos_weight=pw)
    e_loss = F.binary_cross_entropy_with_logits(
        end_logits.squeeze(-1)[mask], end_tgt[mask], pos_weight=pw)
    return s_loss + e_loss


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    # pos_weight scalar: --pos-weight overrides --pos-weight-path (legacy npy)
    if args.pos_weight is not None:
        pos_weight_bin = torch.tensor([args.pos_weight], dtype=torch.float32,
                                      device=device)
    else:
        pos_weight_bin = torch.tensor(
            np.load(args.pos_weight_path).astype(np.float32), device=device)

    ap_best = 0.0   # best avg_mAP (class-agnostic)
    start_epoch = 0

    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print('checkpoint info: epoch', start_epoch + 1, 'avg_mAP', ap_best)

    os.makedirs('final_model', exist_ok=True)

    for e in range(start_epoch, args.max_epoch):
        lam1, lam2 = get_lambda(e, args.phase1_epochs, args.phase2_epochs,
                                args.lambda1, args.lambda2)
        model.train()
        sum_bce_v = sum_nce = sum_cts = sum_fbce = sum_p3 = sum_ctr = sum_bnd = 0.0
        n_iters = min(len(normal_loader), len(anomaly_loader))
        t_start = time.time()
        pbar = tqdm(range(n_iters),
                    desc=f'Ep {e+1}/{args.max_epoch}',
                    disable=not sys.stderr.isatty(), leave=False)

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in pbar:
            n_feat, n_lab, n_ybin, n_len = next(normal_iter)
            a_feat, a_lab, a_ybin, a_len = next(anomaly_iter)

            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            y_bin = torch.cat([n_ybin, a_ybin], dim=0).to(device)        # [2B, T]
            text_labels = list(n_lab) + list(a_lab)
            lengths = torch.cat([n_len, a_len], dim=0).to(device)
            text_labels_t = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2, start_logits, end_logits = model(visual, None, prompt_text, lengths)

            # Original MIL losses (preserved)
            loss_bce_v = CLAS2(logits1, text_labels_t, lengths, device)
            loss_nce   = CLASM(logits2, text_labels_t, lengths, device)

            # Text feature divergence (unchanged structure)
            loss_cts = torch.zeros(1).to(device)
            tf_n = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                tf_a = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss_cts += torch.abs(tf_n @ tf_a)
            loss_cts = loss_cts / 13 * 1e-1

            # New losses (phase-gated)
            if lam1 > 0:
                logits1_2d = logits1.squeeze(-1)                          # [2B, T]
                mask_T = (torch.arange(logits1_2d.shape[1], device=device)
                          .unsqueeze(0) < lengths.unsqueeze(1))            # [2B, T]
                if args.focal_gamma > 0:
                    loss_fbce = focal_bce_loss(logits1_2d, y_bin, mask_T,
                                               pos_weight_bin,
                                               gamma=args.focal_gamma)
                else:
                    loss_fbce = frame_bce_loss(logits1_2d, y_bin, mask_T,
                                               pos_weight_bin)
            else:
                loss_fbce = torch.zeros(1, device=device)

            if lam2 > 0 or args.lambda_contrast > 0:
                probs = torch.sigmoid(logits1.squeeze(-1))
                mask_T2 = (torch.arange(probs.shape[1], device=device)
                           .unsqueeze(0) < lengths.unsqueeze(1))
            if lam2 > 0:
                if args.phase3_loss == 'dice':
                    loss_p3 = dice_loss_anomaly(probs, y_bin, mask_T2)
                else:
                    loss_p3 = tv_smoothness_loss(probs, mask_T2)
            else:
                loss_p3 = torch.zeros(1, device=device)

            if lam2 > 0 and args.lambda_contrast > 0:
                loss_ctr = within_video_contrast_loss(
                    probs, y_bin, mask_T2, margin=args.contrast_margin)
            else:
                loss_ctr = torch.zeros(1, device=device)

            # Boundary start/end loss (Phase 2+, same gate as frame BCE)
            if lam1 > 0 and args.lambda_boundary > 0:
                if 'mask_T' not in locals():
                    logits1_2d_ = logits1.squeeze(-1)
                    mask_T = (torch.arange(logits1_2d_.shape[1], device=device)
                              .unsqueeze(0) < lengths.unsqueeze(1))
                loss_bnd = boundary_bce_loss(
                    start_logits, end_logits, y_bin, mask_T,
                    sigma=args.boundary_sigma,
                    pos_weight=args.boundary_pos_weight)
            else:
                loss_bnd = torch.zeros(1, device=device)

            loss = (loss_bce_v + loss_nce + loss_cts
                    + lam1 * loss_fbce + lam2 * loss_p3
                    + args.lambda_contrast * loss_ctr
                    + lam1 * args.lambda_boundary * loss_bnd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_bce_v += float(loss_bce_v)
            sum_nce   += float(loss_nce)
            sum_cts   += float(loss_cts)
            sum_fbce  += float(loss_fbce)
            sum_p3    += float(loss_p3)
            sum_ctr   += float(loss_ctr)
            sum_bnd   += float(loss_bnd)
            pbar.set_postfix(bce_v=sum_bce_v / (i + 1),
                             nce=sum_nce / (i + 1),
                             cts=sum_cts / (i + 1),
                             fbce=sum_fbce / (i + 1),
                             p3=sum_p3 / (i + 1),
                             ctr=sum_ctr / (i + 1))

        train_secs = time.time() - t_start
        avg_bce_v = sum_bce_v / n_iters
        avg_nce = sum_nce / n_iters
        avg_cts = sum_cts / n_iters
        avg_fbce = sum_fbce / n_iters
        avg_p3 = sum_p3 / n_iters
        avg_ctr = sum_ctr / n_iters
        avg_bnd = sum_bnd / n_iters

        # End-of-epoch eval — model selection by class-agnostic avg_mAP
        AUC, avg_mAP, dmap_ag = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, device, quiet=True,
            inference=args.inference,
            bsn_start_thresh=args.bsn_start_thresh,
            bsn_end_thresh=args.bsn_end_thresh,
            bsn_max_dur=args.bsn_max_dur)
        ag_str = '/'.join(f'{v:.2f}' for v in dmap_ag[:5])
        is_best = avg_mAP > ap_best
        tag = ' *' if is_best else ''
        print(f'[ep {e+1:2d}/{args.max_epoch} {train_secs:.0f}s] '
              f'lam=({lam1},{lam2}) | '
              f'bce_v={avg_bce_v:.3f} nce={avg_nce:.3f} cts={avg_cts:.4f} '
              f'fbce={avg_fbce:.3f} p3={avg_p3:.3f} ctr={avg_ctr:.3f} bnd={avg_bnd:.3f} | '
              f'AUC={AUC:.4f} mAP={avg_mAP:.2f} [{ag_str}]{tag}',
              flush=True)
        if is_best:
            ap_best = avg_mAP
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}, args.checkpoint_path)

        scheduler.step()
        torch.save(model.state_dict(), 'final_model/model_cur.pth')

    best_ck = torch.load(args.checkpoint_path, weights_only=False,
                         map_location=device)
    torch.save(best_ck['model_state_dict'], args.model_path)
    print(f'Final best avg_mAP_agnostic = {ap_best:.2f}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    setup_seed(args.seed)

    label_map = LABEL_MAP

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, True,
        json_path=args.train_json)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, False,
        json_path=args.train_json)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
