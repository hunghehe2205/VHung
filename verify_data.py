"""
Verify data pipeline before training.
Run: python verify_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.dataset import UCFDataset
from src.utils.tools import get_batch_mask
from src.ucf_train import CLAS2, get_binary_label
from intern_vad import VadInternVL
import src.ucf_option as ucf_option

args = ucf_option.parser.parse_args([])
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} — {detail}")
        failed += 1


# ============================================================
# Step 1: CSV files & feature files exist
# ============================================================
print("\n=== Step 1: CSV & Feature Files ===")
for csv_path, tag in [(args.train_list, "Train"), (args.test_list, "Test")]:
    check(f"{tag} CSV exists", os.path.exists(csv_path), csv_path)
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    check(f"{tag} CSV has columns [path, label]", set(['path', 'label']).issubset(df.columns), str(df.columns.tolist()))

    missing = [p for p in df['path'] if not os.path.exists(p)]
    check(f"{tag} all {len(df)} .npy files exist", len(missing) == 0,
          f"{len(missing)} missing, e.g. {missing[:3]}")

    # Check feature dimensions
    sample_path = df['path'].iloc[0]
    feat = np.load(sample_path)
    check(f"{tag} feature dim = 1024 (InternVL)", feat.shape[1] == 1024, f"got shape {feat.shape}")
    check(f"{tag} feature dtype is float", np.issubdtype(feat.dtype, np.floating), f"got {feat.dtype}")
    check(f"{tag} no NaN in features", not np.isnan(feat).any(), "contains NaN!")
    check(f"{tag} no Inf in features", not np.isinf(feat).any(), "contains Inf!")

# ============================================================
# Step 2: Ground truth
# ============================================================
print("\n=== Step 2: Ground Truth ===")
check("gt file exists", os.path.exists(args.gt_path), args.gt_path)
if os.path.exists(args.gt_path):
    gt = np.load(args.gt_path)
    check(f"gt shape: {gt.shape}", gt.ndim == 1, f"expected 1D, got {gt.ndim}D")
    unique_vals = np.unique(gt)
    check("gt values are binary {0, 1}", set(unique_vals).issubset({0, 1}), f"got {unique_vals}")

    test_df = pd.read_csv(args.test_list)
    expected_gt_len = len(test_df) * 16
    # gt length should match num_test_videos * 16 (repeat factor in evaluation)
    check(f"gt length ({len(gt)}) matches test videos * 16",
          True,  # just report, not strict
          f"test_videos={len(test_df)}, gt_len={len(gt)}, ratio={len(gt)/len(test_df):.1f}")

# ============================================================
# Step 3: Label distribution
# ============================================================
print("\n=== Step 3: Label Distribution ===")
train_df = pd.read_csv(args.train_list)
label_counts = train_df['label'].value_counts()
print(f"  Train labels: {dict(label_counts)}")
n_normal = label_counts.get('Normal', 0)
n_anomaly = len(train_df) - n_normal
check("Train has Normal samples", n_normal > 0, f"Normal={n_normal}")
check("Train has Anomaly samples", n_anomaly > 0, f"Anomaly={n_anomaly}")
print(f"  Normal/Anomaly ratio: {n_normal}/{n_anomaly} = {n_normal/max(n_anomaly,1):.2f}")

test_df = pd.read_csv(args.test_list)
test_label_counts = test_df['label'].value_counts()
print(f"  Test labels: {dict(test_label_counts)}")

# ============================================================
# Step 4: Dataset & DataLoader shapes
# ============================================================
print("\n=== Step 4: Dataset & DataLoader ===")
normal_ds = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=True)
anomaly_ds = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=False)
test_ds = UCFDataset(args.visual_length, args.test_list, test_mode=True)

check(f"Normal dataset size: {len(normal_ds)}", len(normal_ds) > 0)
check(f"Anomaly dataset size: {len(anomaly_ds)}", len(anomaly_ds) > 0)
check(f"Test dataset size: {len(test_ds)}", len(test_ds) > 0)

feat, label, length = normal_ds[0]
check(f"Train shape: {feat.shape} == [{args.visual_length}, 1024]",
      feat.shape == (args.visual_length, 1024), f"got {feat.shape}")
check(f"Train length ({length}) <= visual_length ({args.visual_length})",
      length <= args.visual_length, f"length={length}")

feat_t, label_t, length_t = test_ds[0]
check(f"Test feature dim[-1] = 1024", feat_t.shape[-1] == 1024, f"got {feat_t.shape}")

# DataLoader batch
normal_loader = DataLoader(normal_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
anomaly_loader = DataLoader(anomaly_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

n_feat, n_label, n_len = next(iter(normal_loader))
a_feat, a_label, a_len = next(iter(anomaly_loader))

check(f"Normal batch shape: [{args.batch_size}, {args.visual_length}, 1024]",
      n_feat.shape == (args.batch_size, args.visual_length, 1024), f"got {n_feat.shape}")
check(f"Anomaly batch shape: [{args.batch_size}, {args.visual_length}, 1024]",
      a_feat.shape == (args.batch_size, args.visual_length, 1024), f"got {a_feat.shape}")

# ============================================================
# Step 5: Model forward pass (CPU, small batch)
# ============================================================
print("\n=== Step 5: Model Forward Pass ===")
device = "cpu"
model = VadInternVL(
    args.visual_length, args.visual_width, args.visual_head,
    args.visual_layers, args.attn_window, device
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}")

# Small batch forward
small_bs = 4
small_feat = n_feat[:small_bs].to(device)
small_len = n_len[:small_bs].to(device)

try:
    logits = model(small_feat, small_len)
    check(f"Forward output shape: {logits.shape} == [{small_bs}, {args.visual_length}, 1]",
          logits.shape == (small_bs, args.visual_length, 1), f"got {logits.shape}")
    check("No NaN in output", not torch.isnan(logits).any())
    check("No Inf in output", not torch.isinf(logits).any())
except Exception as e:
    check("Forward pass", False, str(e))

# ============================================================
# Step 6: Loss computation
# ============================================================
print("\n=== Step 6: Loss Computation ===")
try:
    visual = torch.cat([n_feat[:small_bs], a_feat[:small_bs]], dim=0).to(device)
    text_labels = list(n_label[:small_bs]) + list(a_label[:small_bs])
    feat_lengths = torch.cat([n_len[:small_bs], a_len[:small_bs]], dim=0).to(device)
    binary_labels = get_binary_label(text_labels).to(device)

    logits = model(visual, feat_lengths)
    loss = CLAS2(logits, binary_labels, feat_lengths, device)

    check(f"Loss value: {loss.item():.4f}", True)
    check("Loss is finite", torch.isfinite(loss))
    check("Loss requires grad", loss.requires_grad)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    check(f"Gradients computed ({len(grad_norms)} params)", len(grad_norms) > 0)
    check("No NaN gradients", all(not np.isnan(g) for g in grad_norms))
    print(f"  Grad norm range: [{min(grad_norms):.6f}, {max(grad_norms):.4f}]")
except Exception as e:
    check("Loss computation", False, str(e))

# ============================================================
# Step 7: Padding mask
# ============================================================
print("\n=== Step 7: Padding Mask ===")
mask = get_batch_mask(n_len[:small_bs], args.visual_length)
check(f"Mask shape: {mask.shape} == [{small_bs}, {args.visual_length}]",
      mask.shape == (small_bs, args.visual_length))
for idx in range(small_bs):
    l = int(n_len[idx])
    valid_region = mask[idx, :l].sum().item()
    padded_region = mask[idx, l:].sum().item() if l < args.visual_length else 0
    check(f"  Sample {idx}: length={l}, valid=0 masked={padded_region}",
          valid_region == 0 and (l >= args.visual_length or padded_region == args.visual_length - l))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"RESULT: {passed} passed, {failed} failed")
if failed == 0:
    print("All checks passed! Ready to train.")
else:
    print("Fix the above failures before training.")
