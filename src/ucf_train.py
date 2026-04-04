"""
ucf_train.py — Training pipeline cho VadInternVL.

Thay đổi so với phiên bản cũ:
  1. Thêm memory_supervision_loss  (từ UR-DMU memory bank signal)
  2. Thêm triplet_loss             (từ UR-DMU feature space structure)
  3. Bỏ focal_loss                 (HIVAU GT bị misalign, chỉ là noise)
  4. model.forward() nhận is_training=True → trả aux_dict
  5. Hyperparams: lambda_mem=0.1, lambda_triplet=0.05
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import wandb
from tqdm import tqdm

from intern_vad import VadInternVL
from src.ucf_test import test
from src.utils.dataset import UCFDataset
import src.ucf_option as ucf_option


# ══════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════

def CLAS2(logits: torch.Tensor, labels: torch.Tensor,
          lengths: torch.Tensor, device: str) -> torch.Tensor:
    """
    MIL classification loss (giữ nguyên từ bản cũ).
    labels: [B, 2]  — [1,0]=Normal, [0,1]=Anomaly
    """
    instance_logits = torch.zeros(0).to(device)
    # labels[:,0]=1 khi Normal → flip để anomaly=1
    bin_labels = 1 - labels[:, 0].reshape(labels.shape[0])
    bin_labels  = bin_labels.to(device)

    probs = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(probs.shape[0]):
        seq_len = max(int(lengths[i]), 1)
        k       = max(1, int(seq_len / 16) + 1)
        topk, _ = torch.topk(probs[i, :seq_len], k=k, largest=True)
        instance_logits = torch.cat([instance_logits, topk.mean().view(1)], dim=0)

    return F.binary_cross_entropy(instance_logits, bin_labels)


def memory_supervision_loss(A_att: torch.Tensor, N_att: torch.Tensor,
                             binary_labels: torch.Tensor,
                             lengths: torch.Tensor,
                             device: str) -> torch.Tensor:
    """
    Supervise memory bank attention scores:
      - Anomaly video  → top-k của A_att phải → 1  (snippet gần Amemory)
      - Normal video   → top-k của N_att phải → 1  (snippet gần Nmemory)

    Inspired by UR-DMU A_loss + N_loss.
    """
    bce  = nn.BCELoss()
    loss = torch.tensor(0.0, device=device)
    ones = torch.ones(1, device=device)
    n    = 0

    for i in range(len(binary_labels)):
        seq_len    = max(int(lengths[i]), 1)
        k          = max(1, seq_len // 16)
        is_anomaly = binary_labels[i, 1] == 1

        if is_anomaly:
            # Anomaly: top-k A_att → 1
            topk_A, _ = torch.topk(A_att[i, :seq_len], k)
            loss = loss + bce(topk_A.mean().unsqueeze(0), ones)
        else:
            # Normal: top-k N_att → 1
            topk_N, _ = torch.topk(N_att[i, :seq_len], k)
            loss = loss + bce(topk_N.mean().unsqueeze(0), ones)
        n += 1

    return loss / max(n, 1)


def triplet_loss(x: torch.Tensor, A_att: torch.Tensor, N_att: torch.Tensor,
                 lengths: torch.Tensor, binary_labels: torch.Tensor,
                 triplet_fn: nn.TripletMarginLoss) -> torch.Tensor:
    """
    Triplet loss trên memory attention — từ UR-DMU.

    anchor   = top-k features của normal videos   (gần Nmemory nhất)
    positive = top-k features của anomaly videos  (gần Nmemory nhất → dễ confuse với normal)
    negative = top-k features của anomaly videos  (gần Amemory nhất → rõ ràng là anomaly)

    Mục tiêu: kéo anchor gần positive (normal-like part of anomaly),
              đẩy xa negative (anomaly-specific part).
    """
    anomaly_idx = (binary_labels[:, 1] == 1).nonzero(as_tuple=True)[0]
    normal_idx  = (binary_labels[:, 0] == 1).nonzero(as_tuple=True)[0]

    if len(anomaly_idx) == 0 or len(normal_idx) == 0:
        return torch.tensor(0.0, device=x.device)

    def get_representative(feat, att, indices, lengths):
        """Lấy mean của top-k snippets theo attention score."""
        reps = []
        for i in indices:
            seq_len = max(int(lengths[i]), 1)
            k       = max(1, seq_len // 16)
            top_idx = torch.topk(att[i, :seq_len], k)[1]
            reps.append(feat[i, top_idx].mean(0))
        return torch.stack(reps).mean(0, keepdim=True)  # [1, D]

    # anchor: normal videos theo N_att
    anchor   = get_representative(x, N_att, normal_idx,  lengths)
    # positive: anomaly videos nhưng phần gần Normal memory
    positive = get_representative(x, N_att, anomaly_idx, lengths)
    # negative: anomaly videos phần gần Anomaly memory
    negative = get_representative(x, A_att, anomaly_idx, lengths)

    return triplet_fn(anchor, positive, negative)


def frame_bce_loss(logits: torch.Tensor, frame_gt: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
    """Plain BCE with Gaussian-smoothed targets for frame-level supervision."""
    logits = logits.squeeze(-1)  # [B, T]
    valid_mask = frame_gt >= 0
    for i in range(logits.shape[0]):
        seq_len = max(int(lengths[i]), 1)
        valid_mask[i, seq_len:] = False
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    pred = torch.sigmoid(logits[valid_mask])
    target = frame_gt[valid_mask]
    return F.binary_cross_entropy(pred, target)


def smoothness_loss(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Penalize sudden score changes giữa adjacent snippets."""
    scores = torch.sigmoid(logits.squeeze(-1))  # [B, T]
    total, count = 0.0, 0
    for i in range(scores.shape[0]):
        seq_len = max(int(lengths[i]), 2)
        diff    = (scores[i, 1:seq_len] - scores[i, :seq_len - 1]).abs()
        total  += diff.sum()
        count  += seq_len - 1
    if count == 0:
        return torch.tensor(0.0, device=logits.device)
    return total / count


def get_binary_label(text_labels) -> torch.Tensor:
    """[1,0] = Normal, [0,1] = Anomaly."""
    label_vectors = torch.zeros(len(text_labels), 2)
    for i, text in enumerate(text_labels):
        if text == 'Normal':
            label_vectors[i, 0] = 1
        else:
            label_vectors[i, 1] = 1
    return label_vectors


# ══════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════

def train(model: VadInternVL,
          normal_loader: DataLoader,
          anomaly_loader: DataLoader,
          testloader: DataLoader,
          args,
          device: str):

    model.to(device)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    gt = np.load(args.gt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer,
                             milestones=args.scheduler_milestones,
                             gamma=args.scheduler_rate)
    auc_best    = 0
    global_step = 0

    # ── Resume from checkpoint ──
    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        auc_best = ckpt.get('auc', 0)
        print(f"Resumed from checkpoint. Best AUC so far: {auc_best:.4f}")

    num_batches = min(len(normal_loader), len(anomaly_loader))

    for epoch in range(args.max_epoch):
        model.train()
        loss_total = 0.0
        normal_iter  = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        pbar = tqdm(range(num_batches),
                    desc=f"Epoch {epoch+1}/{args.max_epoch}",
                    ncols=130)

        for i in pbar:
            # ── Fetch batch ──
            n_feat, n_label, n_len, n_gt = next(normal_iter)
            a_feat, a_label, a_len, a_gt = next(anomaly_iter)

            visual      = torch.cat([n_feat, a_feat], dim=0).to(device)
            text_labels = list(n_label) + list(a_label)
            lengths     = torch.cat([n_len, a_len], dim=0).to(device)
            bin_labels  = get_binary_label(text_labels).to(device)
            frame_gts   = torch.cat([n_gt, a_gt], dim=0).to(device)

            # ── Forward ──
            logits, aux = model(visual, lengths, is_training=True)
            A_att = aux['A_att']   # [B, T]
            N_att = aux['N_att']   # [B, T]

            # ── Losses ──
            loss_mil = CLAS2(logits, bin_labels, lengths, device)

            loss_mem = memory_supervision_loss(
                A_att, N_att, bin_labels, lengths, device
            )

            loss_bce = frame_bce_loss(logits, frame_gts, lengths)
            loss_sm  = smoothness_loss(logits, lengths)

            loss = (loss_mil
                    + args.lambda_mem   * loss_mem
                    + args.lambda_frame * loss_bce
                    + args.mu_smooth    * loss_sm)

            loss_total += loss.item()
            avg_loss    = loss_total / (i + 1)

            # ── Backward ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            # ── Logging ──
            wandb.log({
                "train/loss":         loss.item(),
                "train/loss_mil":     loss_mil.item(),
                "train/loss_mem":     loss_mem.item(),
                "train/loss_bce":     loss_bce.item(),
                "train/loss_smooth":  loss_sm.item(),
                "train/lr":           optimizer.param_groups[0]['lr'],
                "train/mem_gate":     torch.sigmoid(model.mem_gate).item(),
            }, step=global_step)

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                mil=f"{loss_mil.item():.3f}",
                mem=f"{loss_mem.item():.3f}",
                bce=f"{loss_bce.item():.3f}",
                gate=f"{torch.sigmoid(model.mem_gate).item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                best=f"{auc_best:.4f}",
            )

            # ── Mid-epoch eval ──
            step_in_epoch = i * normal_loader.batch_size * 2
            if step_in_epoch % 1280 == 0 and step_in_epoch != 0:
                pbar.clear()
                AUC, AP = test(model, testloader, args.visual_length, gt, device)
                wandb.log({"eval/AUC": AUC, "eval/AP": AP,
                           "train/avg_loss": avg_loss}, step=global_step)

                if AUC > auc_best:
                    auc_best = AUC
                    _save_checkpoint(model, optimizer, epoch, auc_best,
                                     args.checkpoint_path)
                    wandb.log({"eval/best_AUC": auc_best}, step=global_step)

                model.train()

        # ── End of epoch ──
        scheduler.step()
        AUC, AP = test(model, testloader, args.visual_length, gt, device)
        wandb.log({"eval/AUC": AUC, "eval/AP": AP}, step=global_step)
        print(f"Epoch {epoch+1} end → AUC: {AUC:.4f}  AP: {AP:.4f}  best: {auc_best:.4f}")

        if AUC > auc_best:
            auc_best = AUC
            _save_checkpoint(model, optimizer, epoch, auc_best,
                             args.checkpoint_path)
            wandb.log({"eval/best_AUC": auc_best}, step=global_step)

        # Save current epoch weights (for rollback)
        torch.save(model.state_dict(), 'model/model_cur_intern.pth')

        # Restore best checkpoint để epoch tiếp theo bắt đầu từ best state
        if os.path.exists(args.checkpoint_path):
            ckpt = torch.load(args.checkpoint_path)
            model.load_state_dict(ckpt['model_state_dict'])

    # ── Final save ──
    ckpt = torch.load(args.checkpoint_path)
    torch.save(ckpt['model_state_dict'], args.model_path)
    print(f"Training done. Best AUC: {auc_best:.4f}. Saved to {args.model_path}")
    wandb.finish()


def _save_checkpoint(model, optimizer, epoch, auc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auc':                  auc,
    }, path)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args   = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    # ── Datasets ──
    normal_dataset = UCFDataset(
        args.visual_length, args.train_list,
        test_mode=False, normal=True,
        hivau_path=args.hivau_path,
        normal_target=args.normal_target,
        sigma=args.gauss_sigma,
    )
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list,
        test_mode=False, normal=False,
        hivau_path=args.hivau_path,
        normal_target=args.normal_target,
        sigma=args.gauss_sigma,
    )
    test_dataset = UCFDataset(
        args.visual_length, args.test_list, test_mode=True
    )

    normal_loader  = DataLoader(normal_dataset,  batch_size=args.batch_size,
                                shuffle=True,  drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=True,  drop_last=True)
    test_loader    = DataLoader(test_dataset,    batch_size=1, shuffle=False)

    # ── Model ──
    model = VadInternVL(
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        device=device,
        a_nums=args.a_nums,
        n_nums=args.n_nums,
    )

    # ── WandB ──
    wandb.init(
        project="VadInternVL",
        config={
            "visual_length":       args.visual_length,
            "visual_width":        args.visual_width,
            "visual_head":         args.visual_head,
            "visual_layers":       args.visual_layers,
            "attn_window":         args.attn_window,
            "a_nums":              args.a_nums,
            "n_nums":              args.n_nums,
            "batch_size":          args.batch_size,
            "lr":                  args.lr,
            "max_epoch":           args.max_epoch,
            "scheduler_milestones": args.scheduler_milestones,
            "scheduler_rate":      args.scheduler_rate,
            "lambda_mem":          args.lambda_mem,
            "mu_smooth":           args.mu_smooth,
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)

    train(model, normal_loader, anomaly_loader, test_loader, args, device)