import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
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
from src.ucf_train import CLAS2, get_binary_label
import src.ucf_option_v2 as ucf_option


def train(model, normal_loader, anomaly_loader, testloader, args, device):
    model.to(device)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    gt = np.load(args.gt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    auc_best = 0

    global_step = 0
    num_batches = min(len(normal_loader), len(anomaly_loader))

    for e in range(args.max_epoch):
        model.train()
        loss_total = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        pbar = tqdm(range(num_batches), desc=f"Epoch {e+1}/{args.max_epoch}", ncols=120)
        for i in pbar:
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            binary_labels = get_binary_label(text_labels).to(device)

            logits = model(visual_features, feat_lengths)

            loss = CLAS2(logits, binary_labels, feat_lengths, device)
            loss_total += loss.item()
            avg_loss = loss_total / (i + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]['lr']}, step=global_step)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}", best_auc=f"{auc_best:.4f}")

            step = i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                pbar.clear()
                AUC, AP = test(model, testloader, args.visual_length, gt, device)
                wandb.log({"eval/AUC": AUC, "eval/AP": AP, "train/avg_loss": avg_loss}, step=global_step)

                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': auc_best,
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                    wandb.log({"eval/best_AUC": auc_best}, step=global_step)

                model.train()

        scheduler.step()

        torch.save(model.state_dict(), 'model/model_cur_intern_v2.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    wandb.finish()


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

    wandb.init(project="VadInternVL", name="v2-head1-lr5e4-win16", config={
        "visual_length": args.visual_length,
        "visual_width": args.visual_width,
        "visual_head": args.visual_head,
        "visual_layers": args.visual_layers,
        "attn_window": args.attn_window,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epoch": args.max_epoch,
        "scheduler_milestones": args.scheduler_milestones,
        "scheduler_rate": args.scheduler_rate,
    })
    wandb.watch(model, log="gradients", log_freq=100)

    train(model, normal_loader, anomaly_loader, test_loader, args, device)
