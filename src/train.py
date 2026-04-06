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


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

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
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)

    os.makedirs('final_model', exist_ok=True)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_iters = min(len(normal_loader), len(anomaly_loader))
        pbar = tqdm(range(num_iters), desc=f'Epoch {e+1}/{args.max_epoch}')

        eval_every = 20  # evaluate every 20 iterations
        for i in pbar:
            normal_features, normal_label, normal_lengths, normal_gt = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_gt = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            frame_gt = torch.cat([normal_gt, anomaly_gt], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            # Text feature divergence loss
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            # Supervised focal loss from HIVAU annotations (anomaly videos only)
            loss_sup = torch.zeros(1).to(device)
            if args.lambda_sup > 0:
                logits1_flat = logits1.squeeze(-1)
                batch_size = logits1_flat.shape[0]
                half = batch_size // 2  # first half = normal, second half = anomaly
                count = 0
                for k in range(half, batch_size):
                    length_k = feat_lengths[k]
                    pred = torch.sigmoid(logits1_flat[k, :length_k])
                    target = frame_gt[k, :length_k]
                    # Focal loss: -alpha * (1-pt)^gamma * log(pt)
                    pt = pred * target + (1 - pred) * (1 - target)
                    focal_weight = (1 - pt) ** args.focal_gamma
                    alpha = target * args.focal_alpha + (1 - target) * (1 - args.focal_alpha)
                    bce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
                    loss_sup += (alpha * focal_weight * bce).mean()
                    count += 1
                if count > 0:
                    loss_sup = loss_sup / count

            loss = loss1 + loss2 + loss3 + args.lambda_sup * loss_sup

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(loss1=loss_total1 / (i + 1),
                             loss2=loss_total2 / (i + 1),
                             loss3=loss3.item(),
                             loss_sup=loss_sup.item())

            if (i + 1) % eval_every == 0:
                AUC, AP = test(model, testloader, args.visual_length, prompt_text,
                               gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()
        torch.save(model.state_dict(), 'final_model/model_cur.pth')

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    setup_seed(args.seed)

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, True,
                                args.hivau_json, args.boundary_margin, args.label_smooth)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, False,
                                 args.hivau_json, args.boundary_margin, args.label_smooth)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, LABEL_MAP, device)
