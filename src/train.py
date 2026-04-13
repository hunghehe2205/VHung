import os
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
import option


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')
    logger = logging.getLogger('logits3')
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


def map_bce_loss(logits3, soft_mask, lengths):
    raw = logits3.squeeze(-1)
    total_loss = 0.0
    total_count = 0
    for i in range(raw.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        total_loss += F.binary_cross_entropy_with_logits(raw[i, :L], soft_mask[i, :L], reduction='sum')
        total_count += L
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count


def map_smooth_loss(logits3, raw_mask, lengths):
    scores = torch.sigmoid(logits3.squeeze(-1))
    total_loss = 0.0
    total_count = 0
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L < 2:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        same_label = (m[:-1] - m[1:]).abs() < 0.5
        diffs = (s[:-1] - s[1:]).abs()
        masked_diffs = diffs * same_label.float()
        total_loss += masked_diffs.sum()
        total_count += same_label.sum().item()
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count


def map_coverage_loss(logits3, raw_mask, lengths, threshold=0.5):
    """Push anomaly frame scores above threshold."""
    scores = torch.sigmoid(logits3.squeeze(-1))
    total_loss = 0.0
    total_count = 0
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        anom_mask = m > 0.5
        if anom_mask.any():
            shortfall = F.relu(threshold - s[anom_mask])
            total_loss += shortfall.sum()
            total_count += anom_mask.sum().item()
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count


def map_ranking_loss(logits3, raw_mask, lengths, margin=0.3):
    """Ensure max(anomaly) - max(normal) > margin per sample."""
    scores = torch.sigmoid(logits3.squeeze(-1))
    total_loss = 0.0
    total_count = 0
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        anom_mask = m > 0.5
        norm_mask = m < 0.5
        if anom_mask.any() and norm_mask.any():
            max_anom = s[anom_mask].max()
            max_norm = s[norm_mask].max()
            total_loss += F.relu(margin - (max_anom - max_norm))
            total_count += 1
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count


def compute_map_stats(logits3, raw_mask, lengths):
    scores = torch.sigmoid(logits3.squeeze(-1)).detach()
    anomaly_scores = []
    normal_scores = []
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        if (m > 0.5).any():
            anomaly_scores.append(s[m > 0.5])
        if (m < 0.5).any():
            normal_scores.append(s[m < 0.5])

    stats = {}
    if anomaly_scores:
        all_anom = torch.cat(anomaly_scores)
        stats['anomaly_mean'] = all_anom.mean().item()
        stats['coverage'] = (all_anom > 0.5).float().mean().item()
    else:
        stats['anomaly_mean'] = 0.0
        stats['coverage'] = 0.0
    if normal_scores:
        stats['normal_mean'] = torch.cat(normal_scores).mean().item()
    else:
        stats['normal_mean'] = 0.0
    stats['gap'] = stats['anomaly_mean'] - stats['normal_mean']
    return stats


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    logger = setup_logging(args.log_dir)
    log_metrics(logger, f"=== Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_metrics(logger, f"lambda_bce={args.lambda_bce} lambda_smooth={args.lambda_smooth} "
                        f"lambda_cov={args.lambda_coverage} lambda_rank={args.lambda_ranking} "
                        f"smooth_sigma={args.smooth_sigma} lr={args.lr} epochs={args.max_epoch} "
                        f"detach=False")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        log_metrics(logger, f"Loaded checkpoint: epoch={epoch + 1} ap={ap_best}")

    os.makedirs('final_model', exist_ok=True)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_bce = 0
        loss_total_smooth = 0
        loss_total_cov = 0
        loss_total_rank = 0
        epoch_map_stats = {'anomaly_mean': 0, 'normal_mean': 0, 'gap': 0, 'coverage': 0}
        stat_count = 0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_iters = min(len(normal_loader), len(anomaly_loader))
        pbar = tqdm(range(num_iters), desc=f'Epoch {e+1}/{args.max_epoch}')

        for i in pbar:
            step = 0
            normal_features, normal_label, normal_lengths, normal_soft_mask, normal_raw_mask = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_soft_mask, anomaly_raw_mask = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            soft_mask_batch = torch.cat([normal_soft_mask, anomaly_soft_mask], dim=0).to(device)
            raw_mask_batch = torch.cat([normal_raw_mask, anomaly_raw_mask], dim=0).to(device)

            text_features, logits1, logits2, logits3 = model(visual_features, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1
            loss_total3 += loss3.item()

            l_bce = map_bce_loss(logits3, soft_mask_batch, feat_lengths)
            l_smooth = map_smooth_loss(logits3, raw_mask_batch, feat_lengths)
            l_cov = map_coverage_loss(logits3, raw_mask_batch, feat_lengths, threshold=args.coverage_threshold)
            l_rank = map_ranking_loss(logits3, raw_mask_batch, feat_lengths, margin=args.ranking_margin)
            loss_total_bce += l_bce.item()
            loss_total_smooth += l_smooth.item()
            loss_total_cov += l_cov.item()
            loss_total_rank += l_rank.item()

            loss = (loss1 + loss2 + loss3
                    + args.lambda_bce * l_bce + args.lambda_smooth * l_smooth
                    + args.lambda_coverage * l_cov + args.lambda_ranking * l_rank)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_stats = compute_map_stats(logits3, raw_mask_batch, feat_lengths)
            for k in epoch_map_stats:
                epoch_map_stats[k] += batch_stats[k]
            stat_count += 1

            pbar.set_postfix(loss1=loss_total1 / (i + 1),
                             loss2=loss_total2 / (i + 1),
                             L_bce=loss_total_bce / (i + 1),
                             L_cov=loss_total_cov / (i + 1),
                             L_rk=loss_total_rank / (i + 1))

            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                n_cur = i + 1
                log_metrics(logger,
                    f"[Epoch {e+1} step={step}] "
                    f"loss1={loss_total1/n_cur:.4f} loss2={loss_total2/n_cur:.4f} loss3={loss_total3/n_cur:.4f} | "
                    f"L_bce={loss_total_bce/n_cur:.4f} L_smooth={loss_total_smooth/n_cur:.4f} "
                    f"L_cov={loss_total_cov/n_cur:.4f} L_rank={loss_total_rank/n_cur:.4f}")

                auc1, _, auc3, _, score_maps = test(model, testloader, args.visual_length, prompt_text,
                                                    gt, gtsegments, gtlabels, device, logger)

                if auc1 > ap_best:
                    ap_best = auc1
                    log_metrics(logger, f"  >> New best AUC1={ap_best:.4f} (AUC3={auc3:.4f})")
                    maps_dir = os.path.join(args.log_dir, 'score_maps')
                    os.makedirs(maps_dir, exist_ok=True)
                    for name, smap in score_maps.items():
                        np.save(os.path.join(maps_dir, f"{name}.npy"), smap)
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                else:
                    log_metrics(logger, f"  Best AUC1={ap_best:.4f} (current={auc1:.4f})")

        n = max(num_iters, 1)
        sc = max(stat_count, 1)
        for k in epoch_map_stats:
            epoch_map_stats[k] /= sc

        log_metrics(logger,
            f"[Epoch {e+1}/{args.max_epoch}] "
            f"loss1={loss_total1/n:.4f} loss2={loss_total2/n:.4f} loss3={loss_total3/n:.4f} | "
            f"L_bce={loss_total_bce/n:.4f} L_smooth={loss_total_smooth/n:.4f} "
            f"L_cov={loss_total_cov/n:.4f} L_rank={loss_total_rank/n:.4f}")
        log_metrics(logger,
            f"  logits3: anomaly={epoch_map_stats['anomaly_mean']:.3f} "
            f"normal={epoch_map_stats['normal_mean']:.3f} "
            f"gap={epoch_map_stats['gap']:.3f} "
            f"coverage={epoch_map_stats['coverage']:.3f}")

        scheduler.step()
        torch.save(model.state_dict(), 'final_model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    log_metrics(logger, f"=== Training finished. Best AUC1: {ap_best:.4f} ===")


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
                                hivau_json_path=args.hivau_json_path, smooth_sigma=args.smooth_sigma)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False,
                                 hivau_json_path=args.hivau_json_path, smooth_sigma=args.smooth_sigma)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
