import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CLIPVAD
from test import LABEL_MAP, test
from utils.dataset import UCFDataset
from utils.tools import get_batch_label, get_prompt_text
import option


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(
            logits[i, 0:lengths[i]],
            k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(
        torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device, y_bin=None):
    """Video-level MIL BCE on logits1 with Exp 18 top-k inside-GT for anomaly."""
    instance_logits = torch.zeros(0).to(device)
    vid_label = 1 - labels[:, 0].reshape(labels.shape[0])
    vid_label = vid_label.to(device)
    probs = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        L = int(lengths[i])
        v = probs[i, :L]
        if y_bin is not None and vid_label[i] > 0.5:
            inside = y_bin[i, :L] > 0.5
            n_inside = int(inside.sum())
            if n_inside > 0:
                pool, k = v[inside], max(1, n_inside // 16 + 1)
            else:
                pool, k = v, max(1, L // 16 + 1)
        else:
            pool, k = v, max(1, L // 16 + 1)
        tmp, _ = torch.topk(pool, k=min(k, pool.shape[0]), largest=True)
        instance_logits = torch.cat([instance_logits, tmp.mean().view(1)], dim=0)
    return F.binary_cross_entropy(instance_logits, vid_label)


def tcn_bce(logits, target, mask, pos_weight):
    """Frame-level BCE on TCN logits [B,T] with Gaussian-smoothed target."""
    logits_m = logits[mask]
    target_m = target[mask]
    return F.binary_cross_entropy_with_logits(
        logits_m, target_m, pos_weight=pos_weight.to(logits_m.device))


def tcn_dice(probs, target, mask, eps=1.0):
    """Anomaly-only soft Dice on sigmoid(tcn_logits)."""
    mask_f = mask.float()
    p = probs * mask_f
    y = target * mask_f
    inter = (p * y).sum(-1)
    denom = p.sum(-1) + y.sum(-1)
    dice = (2.0 * inter + eps) / (denom + eps)
    has_pos = (y.sum(-1) > 0).float()
    return ((1.0 - dice) * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def tcn_ctr(probs, target, mask, margin=0.3):
    """Within-video contrast on sigmoid(tcn_logits). Anomaly-only."""
    valid = mask.float()
    inside = (target > 0.5).float() * valid
    outside = (1.0 - (target > 0.5).float()) * valid
    has_inside = (inside.sum(-1) > 0).float()
    inside_mean = (probs * inside).sum(-1) / inside.sum(-1).clamp_min(1.0)
    outside_mean = (probs * outside).sum(-1) / outside.sum(-1).clamp_min(1.0)
    gap_loss = F.relu(margin - (inside_mean - outside_mean))
    return (gap_loss * has_inside).sum() / has_inside.sum().clamp_min(1.0)


def get_lambdas(epoch, phase1_epochs, phase2_epochs):
    """3-phase curriculum for TCN losses.
    P1: CLAS2+CLASM warmup; P2: +tcn_bce; P3: +tcn_dice, +tcn_ctr."""
    if epoch < phase1_epochs:
        return 0.0, 0.0, 0.0, 1
    elif epoch < phase2_epochs:
        return 1.0, 0.0, 0.0, 2
    else:
        return 1.0, 1.0, 1.0, 3


def _split_param_groups(model, lr_backbone, lr_tcn):
    tcn_params = [p for n, p in model.named_parameters()
                  if n.startswith('tcn.') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('tcn.') and p.requires_grad]
    return [
        {'params': other_params, 'lr': lr_backbone, 'weight_decay': 0.0},
        {'params': tcn_params, 'lr': lr_tcn, 'weight_decay': 0.0},
    ]


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(_split_param_groups(model, args.lr, args.lr_tcn))
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    pos_weight_tcn = torch.tensor([args.tcn_pos_weight],
                                  dtype=torch.float32, device=device)

    ap_best_all = 0.0
    ap_best_abn = 0.0
    start_epoch = 0
    if args.use_checkpoint:
        ckpt = torch.load(args.checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        ap_best_all = ckpt.get('ap_all', ckpt.get('ap', 0.0))
        ap_best_abn = ckpt.get('ap_abn', ckpt.get('ap', 0.0))

    os.makedirs('final_model', exist_ok=True)

    prev_total = None
    for e in range(start_epoch, args.max_epoch):
        lam_bce, lam_dice, lam_ctr, phase = get_lambdas(
            e, args.phase1_epochs, args.phase2_epochs)

        if e == args.phase1_epochs:
            print(f'[curriculum ep {e + 1} P1->P2] activating: tcn_bce (λ=1.0)',
                  flush=True)
        if e == args.phase2_epochs:
            print(f'[curriculum ep {e + 1} P2->P3] activating: '
                  f'tcn_dice (λ=1.0), tcn_ctr (λ=1.0)', flush=True)

        model.train()
        sum_clas2 = sum_clasm = sum_cts = 0.0
        sum_tbce = sum_tdice = sum_tctr = 0.0
        n_iters = min(len(normal_loader), len(anomaly_loader))
        t_start = time.time()
        pbar = tqdm(range(n_iters), desc=f'Ep {e + 1}/{args.max_epoch} P{phase}',
                    disable=not sys.stderr.isatty(), leave=False)
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in pbar:
            n_feat, n_lab, n_ybin, n_ysoft, n_len = next(normal_iter)
            a_feat, a_lab, a_ybin, a_ysoft, a_len = next(anomaly_iter)

            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            y_bin = torch.cat([n_ybin, a_ybin], dim=0).to(device)
            y_soft = torch.cat([n_ysoft, a_ysoft], dim=0).to(device)
            text_labels = list(n_lab) + list(a_lab)
            lengths = torch.cat([n_len, a_len], dim=0).to(device)
            text_labels_t = get_batch_label(
                text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2, tcn_logits = model(
                visual, None, prompt_text, lengths)

            loss_clas2 = CLAS2(logits1, text_labels_t, lengths, device, y_bin=y_bin)
            if logits2 is not None:
                loss_clasm = CLASM(logits2, text_labels_t, lengths, device)
            else:
                loss_clasm = torch.zeros(1, device=device)

            loss_cts = torch.zeros(1, device=device)
            if text_features is not None:
                tf_n = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
                for j in range(1, text_features.shape[0]):
                    tf_a = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                    loss_cts = loss_cts + torch.abs(tf_n @ tf_a)
                loss_cts = loss_cts / 13 * args.lambda_cts

            tcn_2d = tcn_logits.squeeze(-1)
            mask_T = (torch.arange(tcn_2d.shape[1], device=device)
                      .unsqueeze(0) < lengths.unsqueeze(1))

            if lam_bce > 0:
                loss_tbce = tcn_bce(tcn_2d, y_soft, mask_T, pos_weight_tcn)
            else:
                loss_tbce = torch.zeros(1, device=device)

            probs = None
            if lam_dice > 0 or lam_ctr > 0:
                probs = torch.sigmoid(tcn_2d)
            eff_lam_dice = lam_dice * args.lambda_tcn_dice
            eff_lam_ctr = lam_ctr * args.lambda_tcn_ctr
            if eff_lam_dice > 0:
                loss_tdice = tcn_dice(probs, y_bin, mask_T)
            else:
                loss_tdice = torch.zeros(1, device=device)
            if eff_lam_ctr > 0:
                loss_tctr = tcn_ctr(probs, y_bin, mask_T, margin=args.contrast_margin)
            else:
                loss_tctr = torch.zeros(1, device=device)

            loss = (args.lambda_clas2 * loss_clas2
                    + args.lambda_nce * loss_clasm
                    + loss_cts
                    + lam_bce * loss_tbce
                    + eff_lam_dice * loss_tdice
                    + eff_lam_ctr * loss_tctr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_clas2 += float(loss_clas2)
            sum_clasm += float(loss_clasm)
            sum_cts += float(loss_cts)
            sum_tbce += float(loss_tbce)
            sum_tdice += float(loss_tdice)
            sum_tctr += float(loss_tctr)
            pbar.set_postfix(CLAS2=sum_clas2 / (i + 1),
                             CLASM=sum_clasm / (i + 1),
                             tbce=sum_tbce / (i + 1),
                             tdice=sum_tdice / (i + 1),
                             tctr=sum_tctr / (i + 1))

        train_secs = time.time() - t_start
        avg_clas2 = sum_clas2 / n_iters
        avg_clasm = sum_clasm / n_iters
        avg_cts = sum_cts / n_iters
        avg_tbce = sum_tbce / n_iters
        avg_tdice = sum_tdice / n_iters
        avg_tctr = sum_tctr / n_iters
        total = (args.lambda_clas2 * avg_clas2 + args.lambda_nce * avg_clasm + avg_cts
                 + lam_bce * avg_tbce
                 + (lam_dice * args.lambda_tcn_dice) * avg_tdice
                 + (lam_ctr * args.lambda_tcn_ctr) * avg_tctr)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_tcn = optimizer.param_groups[1]['lr']

        AUC_wsv, avg_mAP_abn, dmap_abn, AUC_tcn, diag = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, device, quiet=True,
            return_diag=True, eval_head=args.eval_head)
        abn_str = '/'.join(f'{v:.2f}' for v in dmap_abn[:5])
        curr_all = diag['mAP_all']
        curr_abn = avg_mAP_abn
        # Best by mAP_all, tie-break by mAP_abn.
        is_best = (curr_all > ap_best_all
                   or (curr_all == ap_best_all and curr_abn > ap_best_abn))
        tag = ' *' if is_best else ''

        print(f'[ep {e + 1:2d}/{args.max_epoch} P{phase} {train_secs:.0f}s] '
              f'lam=({args.lambda_clas2},{args.lambda_nce},{args.lambda_cts},{lam_bce},{lam_dice},{lam_ctr}) '
              f'lr_bb={lr_bb:.1e} lr_tcn={lr_tcn:.1e}', flush=True)
        print(f'  loss: CLAS2={avg_clas2:.3f} CLASM={avg_clasm:.3f} '
              f'cts={avg_cts:.4f} tcn_bce={avg_tbce:.3f} '
              f'tcn_dice={avg_tdice:.3f} tcn_ctr={avg_tctr:.3f} '
              f'total={total:.3f}', flush=True)
        print(f'[ep {e + 1:2d} eval] mAP_abn={avg_mAP_abn:.2f} '
              f'mAP_all={diag["mAP_all"]:.2f} '
              f'AUC_tcn={AUC_tcn:.4f} AUC_wsv={AUC_wsv:.4f} '
              f'bsh_med={diag["bsh_med"]:.4f} '
              f'peak_in_gt={diag["peak_in_gt"]:.3f} '
              f'over_cov_med={diag["over_cov_med"]:.2f}x '
              f'[{abn_str}]{tag}', flush=True)

        if e == args.phase1_epochs and prev_total is not None:
            print(f'[ep {e + 1} P1->P2 loss trace] Δ_total={total - prev_total:+.3f} '
                  f'(expected from tcn_bce activation)', flush=True)
        if e == args.phase2_epochs and prev_total is not None:
            print(f'[ep {e + 1} P2->P3 loss trace] Δ_total={total - prev_total:+.3f} '
                  f'(expected from tcn_dice + tcn_ctr activation)', flush=True)

        frac_high = diag.get('frac_tcn_high', 1.0)
        if frac_high < 0.02:
            print(f'[watchdog] WARN frac(tcn_prob>0.5)={frac_high:.4f} <2% — possible collapse',
                  flush=True)
        if e >= 5 and AUC_tcn < 0.80:
            print(f'[watchdog] WARN AUC_tcn={AUC_tcn:.4f} <0.80 at ep {e + 1}',
                  flush=True)

        if is_best:
            ap_best_all = curr_all
            ap_best_abn = curr_abn
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap_all': ap_best_all,
                        'ap_abn': ap_best_abn}, args.checkpoint_path)

        scheduler.step()
        prev_total = total

    torch.save(model.state_dict(), 'final_model/model_final.pth')
    best_ck = torch.load(args.checkpoint_path, weights_only=False, map_location=device)
    torch.save(best_ck['model_state_dict'], args.model_path)
    print(f'Final best mAP_all = {ap_best_all:.2f} | mAP_abn = {ap_best_abn:.2f}',
          flush=True)


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
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
                    tcn_dilations=tuple(args.tcn_dilations),
                    tcn_input=args.tcn_input,
                    use_a_branch=bool(args.use_a_branch),
                    tcn_multiscale=bool(args.tcn_multiscale))

    if args.load_baseline and os.path.exists(args.load_baseline):
        base = torch.load(args.load_baseline, weights_only=False, map_location=device)
        missing, unexpected = model.load_state_dict(base, strict=False)
        print(f'[load_baseline] missing={len(missing)} unexpected={len(unexpected)}',
              flush=True)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
