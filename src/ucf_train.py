import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from intern_vad import VadInternVL
from src.ucf_test import test
from src.utils.dataset import UCFDataset
import src.ucf_option as ucf_option


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


def get_binary_label(text_labels):
    """Convert text labels to binary: [1,0] for Normal, [0,1] for Anomaly."""
    label_vectors = torch.zeros(len(text_labels), 2)
    for i, text in enumerate(text_labels):
        if text == 'Normal':
            label_vectors[i, 0] = 1
        else:
            label_vectors[i, 1] = 1
    return label_vectors


def train(model, normal_loader, anomaly_loader, testloader, args, device):
    model.to(device)
    gt = np.load(args.gt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    auc_best = 0
    epoch = 0

    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        auc_best = checkpoint['auc']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " auc:", auc_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            binary_labels = get_binary_label(text_labels).to(device)

            logits = model(visual_features, feat_lengths)

            loss = CLAS2(logits, binary_labels, feat_lengths, device)
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e + 1, '| step: ', step, '| loss: ', loss_total / (i + 1))
                AUC, AP = test(model, testloader, args.visual_length, gt, device)

                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': auc_best,
                    }
                    torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()

        torch.save(model.state_dict(), 'model/model_cur_intern.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, normal=True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, normal=False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VadInternVL(
        args.visual_length, args.visual_width, args.visual_head,
        args.visual_layers, args.attn_window, device
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, device)
