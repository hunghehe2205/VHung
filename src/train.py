import os
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


def get_lambda(epoch, phase1_epochs, phase2_epochs, lambda1, lambda2):
    """3-phase schedule for extra losses.
    epoch < phase1_epochs:                return (0, 0)       # MIL-only warmup
    phase1_epochs <= epoch < phase2_epochs: return (lambda1, 0) # +frame BCE
    epoch >= phase2_epochs:                return (lambda1, lambda2) # +IoU
    """
    if epoch < phase1_epochs:
        return 0.0, 0.0
    elif epoch < phase2_epochs:
        return float(lambda1), 0.0
    else:
        return float(lambda1), float(lambda2)


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    # Load pos_weight scalar (precomputed once by compute_pos_weight.py)
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
        sum_bce_v = sum_nce = sum_cts = sum_fbce = sum_iou = 0.0
        n_iters = min(len(normal_loader), len(anomaly_loader))
        pbar = tqdm(range(n_iters),
                    desc=f'Ep {e+1}/{args.max_epoch} lam1={lam1} lam2={lam2}')

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

            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)

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
                loss_fbce = frame_bce_loss(logits1_2d, y_bin, mask_T, pos_weight_bin)
            else:
                loss_fbce = torch.zeros(1, device=device)

            if lam2 > 0:
                probs = torch.sigmoid(logits1.squeeze(-1))
                mask_T2 = (torch.arange(probs.shape[1], device=device)
                           .unsqueeze(0) < lengths.unsqueeze(1))
                loss_iou = soft_iou_loss(probs, y_bin, mask_T2)
            else:
                loss_iou = torch.zeros(1, device=device)

            loss = (loss_bce_v + loss_nce + loss_cts
                    + lam1 * loss_fbce + lam2 * loss_iou)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_bce_v += float(loss_bce_v)
            sum_nce   += float(loss_nce)
            sum_cts   += float(loss_cts)
            sum_fbce  += float(loss_fbce)
            sum_iou   += float(loss_iou)
            pbar.set_postfix(bce_v=sum_bce_v / (i + 1),
                             nce=sum_nce / (i + 1),
                             cts=sum_cts / (i + 1),
                             fbce=sum_fbce / (i + 1),
                             iou=sum_iou / (i + 1))

        # End-of-epoch eval — model selection by class-agnostic avg_mAP
        AUC, avg_mAP = test(model, testloader, args.visual_length, prompt_text,
                            gt, gtsegments, gtlabels, device)
        print(f'[epoch {e+1}] AUC={AUC:.4f}  avg_mAP_agnostic={avg_mAP:.2f}')
        if avg_mAP > ap_best:
            ap_best = avg_mAP
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}, args.checkpoint_path)
            print(f'  -> new best ({avg_mAP:.2f}), checkpoint saved')

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
