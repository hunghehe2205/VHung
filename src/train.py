import math
import os
import time
import logging
from datetime import datetime

import torch
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
from losses_dbranch import dense_loss_total
import option


# ---------- logging ----------

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')
    logger = logging.getLogger('dbranch')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return logger


def log_metrics(logger, msg):
    print(msg)
    logger.info(msg)


# ---------- C/A-branch losses (from VadCLIP, unchanged) ----------

def CLASM(logits, labels, lengths, device):
    """Softmax MIL for 14-class video-level classification (logits2 head)."""
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device):
    """Sigmoid MIL for binary anomaly video-level classification (logits1 head)."""
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    return F.binary_cross_entropy(instance_logits, labels)


# ---------- Schedule C: layered LR + beta ramp ----------

def build_param_groups(model: CLIPVAD, backbone_lr: float, ca_lr: float, dbranch_lr: float):
    """Three groups: backbone (LGT-Adapter), C/A-branch heads, D-Branch.

    CLIP parameters are frozen in CLIPVAD.__init__ and are skipped here too.
    """
    backbone_modules = [
        model.temporal, model.gc1, model.gc2, model.gc3, model.gc4,
        model.linear, model.frame_position_embeddings,
    ]
    ca_modules = [model.mlp1, model.mlp2, model.classifier, model.text_prompt_embeddings]

    def collect(modules):
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    return [
        {'params': list(collect(backbone_modules)), 'lr': backbone_lr, 'name': 'backbone'},
        {'params': list(collect(ca_modules)),       'lr': ca_lr,       'name': 'ca_branch'},
        {'params': list(model.d_branch.parameters()),'lr': dbranch_lr,  'name': 'd_branch'},
    ]


def beta_for_epoch(epoch: int, warmup: int, ramp: int, beta_max: float) -> float:
    """Linear ramp: 0 for epoch < warmup, ramp 0->beta_max over `ramp` epochs."""
    if epoch < warmup:
        return 0.0
    if epoch >= warmup + ramp:
        return beta_max
    return beta_max * (epoch - warmup + 1) / (ramp + 1)


# ---------- main training loop ----------

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    logger = setup_logging(args.log_dir)
    log_metrics(logger,
        f"=== D-Branch training | {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"epochs={args.max_epoch} bs={args.batch_size} | "
        f"backbone_lr={args.backbone_lr} ca_lr={args.lr} dbranch_lr={args.dbranch_lr} ===")
    log_metrics(logger,
        f"[Beta] warmup={args.beta_warmup_epochs} ramp={args.beta_ramp_epochs} beta_max={args.beta_max}")
    log_metrics(logger,
        f"[Loss] w_bce={args.w_bce} w_margin={args.w_margin} w_dice={args.w_dice} w_var={args.w_var} "
        f"margin_m={args.margin_m} margin_temp={args.margin_temp}")

    param_groups = build_param_groups(model, args.backbone_lr, args.lr, args.dbranch_lr)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    best_map_avg = 0.0
    epoch_start = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, KeyError) as e:
                log_metrics(logger, f"[ckpt] optimizer state mismatch ({e}); using fresh optimizer.")
        epoch_start = ckpt.get('epoch', -1) + 1
        best_map_avg = ckpt.get('map_avg', 0.0)
        log_metrics(logger, f"Resumed from checkpoint: next_epoch={epoch_start} best_map_avg={best_map_avg}")

    os.makedirs('final_model', exist_ok=True)

    for e in range(epoch_start, args.max_epoch):
        model.train()
        beta = beta_for_epoch(e, args.beta_warmup_epochs, args.beta_ramp_epochs, args.beta_max)
        epoch_t0 = time.time()

        sum_l1 = sum_l2 = sum_l3 = 0.0
        sum_bce = sum_margin = sum_dice = sum_var = 0.0
        sum_total = 0.0
        sqsum_total = 0.0
        grad_norm_last = 0.0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_iters = min(len(normal_loader), len(anomaly_loader))
        pbar = tqdm(range(num_iters), desc=f'Epoch {e+1}/{args.max_epoch} beta={beta:.3f}')

        for i in pbar:
            normal_features, normal_label, normal_lengths, _, normal_raw_mask = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, _, anomaly_raw_mask = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # Hard binary frame labels (raw mask is 0/1 from events_to_clip_mask).
            y_t = torch.cat([normal_raw_mask, anomaly_raw_mask], dim=0).to(device)

            text_features, logits1, logits2, s_t = model(visual_features, None, prompt_text, feat_lengths)

            # C/A-branch losses (regularize backbone).
            l1 = CLAS2(logits1, text_labels, feat_lengths, device)
            l2 = CLASM(logits2, text_labels, feat_lengths, device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            l3 = torch.zeros(1, device=device)
            for j in range(1, text_features.shape[0]):
                t_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                l3 = l3 + torch.abs(text_feature_normal @ t_abr)
            l3 = l3 / 13 * 1e-1

            loss_ca = l1 + l2 + l3.squeeze()

            # D-Branch dense loss.
            loss_d, parts = dense_loss_total(
                s_t, y_t, feat_lengths,
                w_bce=args.w_bce, w_margin=args.w_margin,
                w_dice=args.w_dice, w_var=args.w_var,
                margin_m=args.margin_m, margin_temp=args.margin_temp,
            )

            loss = loss_ca + beta * loss_d

            if not torch.isfinite(loss):
                log_metrics(logger,
                    f"[FATAL] non-finite loss epoch={e+1} iter={i}: total={loss.item()} "
                    f"l1={l1.item()} l2={l2.item()} l3={l3.item()} "
                    f"bce={parts['bce']} margin={parts['margin']} dice={parts['dice']} var={parts['var']}")
                raise RuntimeError("Training aborted: non-finite loss.")

            optimizer.zero_grad()
            loss.backward()
            grad_norm_last = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')).item()
            optimizer.step()

            sum_l1 += l1.item(); sum_l2 += l2.item(); sum_l3 += l3.item()
            sum_bce += parts['bce']; sum_margin += parts['margin']
            sum_dice += parts['dice']; sum_var += parts['var']
            lv = loss.item(); sum_total += lv; sqsum_total += lv * lv

            pbar.set_postfix(l1=sum_l1/(i+1), l2=sum_l2/(i+1),
                             bce=sum_bce/(i+1), margin=sum_margin/(i+1),
                             dice=sum_dice/(i+1))

        # ---- end of epoch: log + eval + checkpoint ----
        n = max(num_iters, 1)
        mean_total = sum_total / n
        var_total = max(0.0, sqsum_total / n - mean_total ** 2)
        std_total = math.sqrt(var_total)
        epoch_dt = time.time() - epoch_t0

        log_metrics(logger,
            f"[Epoch {e+1}/{args.max_epoch}] beta={beta:.3f} | "
            f"l1={sum_l1/n:.4f} l2={sum_l2/n:.4f} l3={sum_l3/n:.4f} | "
            f"bce={sum_bce/n:.4f} margin={sum_margin/n:.4f} "
            f"dice={sum_dice/n:.4f} var={sum_var/n:.4f} | "
            f"total={mean_total:.4f}±{std_total:.4f} grad={grad_norm_last:.2f} dt={epoch_dt:.1f}s")

        # Evaluate from D-Branch s_t.
        eval_out = test(model, testloader, args.visual_length, prompt_text,
                        gt, gtsegments, gtlabels, device, logger,
                        score_source='dbranch')
        auc = eval_out['auc']; ano_auc = eval_out['ano_auc']
        ap = eval_out['ap']; map_avg = eval_out['map_avg']

        improved = map_avg > best_map_avg
        marker = ">>" if improved else "  "
        log_metrics(logger,
            f"{marker} [Eval e{e+1}] AUC={auc:.4f} AnoAUC={ano_auc:.4f} AP={ap:.4f} "
            f"BinaryMAP_AVG={map_avg:.4f} (best={best_map_avg:.4f})")

        if improved:
            best_map_avg = map_avg
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': auc, 'ano_auc': ano_auc, 'ap': ap, 'map_avg': map_avg,
            }, args.checkpoint_path)

        scheduler.step()
        torch.save(model.state_dict(), 'final_model/model_cur.pth')

    if os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, weights_only=False)
        torch.save(ckpt['model_state_dict'], args.model_path)
    log_metrics(logger,
        f"=== Training finished. Best Binary mAP AVG: {best_map_avg:.4f} ===")


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

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True,
                                hivau_json_path=args.hivau_json_path)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False,
                                 hivau_json_path=args.hivau_json_path)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    # Load VadCLIP pretrained checkpoint to seed backbone + C/A-branch.
    if os.path.exists(args.model_path):
        missing, unexpected = model.load_state_dict(
            torch.load(args.model_path, weights_only=False, map_location=device), strict=False)
        if missing:
            print(f"[load] missing keys (random init): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[load] unexpected keys (ignored): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
