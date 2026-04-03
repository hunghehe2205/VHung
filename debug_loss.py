"""Quick debug: check logits, loss, and gradient flow."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.dataset import UCFDataset
from src.ucf_train import CLAS2, get_binary_label
from intern_vad import VadInternVL
import src.ucf_option as ucf_option

args = ucf_option.parser.parse_args([])
device = "cuda" if torch.cuda.is_available() else "cpu"

normal_ds = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=True)
anomaly_ds = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=False)
normal_loader = DataLoader(normal_ds, batch_size=4, shuffle=True, drop_last=True)
anomaly_loader = DataLoader(anomaly_ds, batch_size=4, shuffle=True, drop_last=True)

model = VadInternVL(
    args.visual_length, args.visual_width, args.visual_head,
    args.visual_layers, args.attn_window, device
).to(device)

n_feat, n_label, n_len = next(iter(normal_loader))
a_feat, a_label, a_len = next(iter(anomaly_loader))

visual = torch.cat([n_feat, a_feat], dim=0).to(device)
text_labels = list(n_label) + list(a_label)
feat_lengths = torch.cat([n_len, a_len], dim=0).to(device)
binary_labels = get_binary_label(text_labels).to(device)

print("=== Input ===")
print(f"visual: {visual.shape}, range: [{visual.min():.4f}, {visual.max():.4f}]")
print(f"feat_lengths: {feat_lengths}")
print(f"labels (text): {text_labels}")
print(f"binary_labels: {binary_labels}")

logits = model(visual, feat_lengths)
print(f"\n=== Raw Logits ===")
print(f"shape: {logits.shape}")
print(f"range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"mean: {logits.mean():.4f}, std: {logits.std():.4f}")

sig = torch.sigmoid(logits)
print(f"\n=== After Sigmoid ===")
print(f"range: [{sig.min():.4f}, {sig.max():.4f}]")
print(f"mean: {sig.mean():.4f}")

# Check per-sample topk values (what CLAS2 uses)
print(f"\n=== Per-sample TopK (what loss sees) ===")
sig_flat = sig.reshape(sig.shape[0], sig.shape[1])
processed_labels = 1 - binary_labels[:, 0]
for i in range(sig_flat.shape[0]):
    seq_len = max(int(feat_lengths[i]), 1)
    topk_vals, _ = torch.topk(sig_flat[i, 0:seq_len], k=int(seq_len / 16 + 1), largest=True)
    topk_mean = topk_vals.mean()
    label = processed_labels[i]
    print(f"  sample {i}: topk_mean={topk_mean:.6f}, label={label:.0f} ({text_labels[i]})")

print(f"\n=== Loss ===")
loss = CLAS2(logits, binary_labels, feat_lengths, device)
print(f"loss: {loss.item():.4f}")

loss.backward()
grad_norms = [(name, p.grad.norm().item()) for name, p in model.named_parameters() if p.grad is not None]
print(f"\n=== Gradient Norms (top 10) ===")
grad_norms.sort(key=lambda x: x[1], reverse=True)
for name, norm in grad_norms[:10]:
    print(f"  {name}: {norm:.6f}")
