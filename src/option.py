import argparse

parser = argparse.ArgumentParser(description='VadCLIP-UCF + D-Branch')
parser.add_argument('--seed', default=234, type=int)

# Model architecture
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

# Training (Schedule C: joint fine-tune + ramp-up beta)
parser.add_argument('--max-epoch', default=12, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Default LR for C/A-branch (logits1 MIL + logits2 text).')
parser.add_argument('--backbone-lr', default=5e-6, type=float,
                    help='LR for shared backbone (Transformer + GCN). '
                         'Lower than --lr to slow drift.')
parser.add_argument('--dbranch-lr', default=1e-4, type=float,
                    help='LR for D-Branch (random init).')
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[6, 10], nargs='+', type=int)

# Beta ramp-up for D-Branch dense loss
parser.add_argument('--beta-max', default=1.0, type=float,
                    help='Max weight applied to total D-Branch loss.')
parser.add_argument('--beta-warmup-epochs', default=3, type=int,
                    help='Epochs of beta=0 (backbone warm-up via C/A-branch only).')
parser.add_argument('--beta-ramp-epochs', default=3, type=int,
                    help='Epochs over which beta linearly ramps 0 -> beta_max.')

# D-Branch loss weights and hyperparameters
parser.add_argument('--w-bce', default=0.3, type=float,
                    help='Anchor / break-symmetry signal; small weight so per-frame '
                         'BCE does not push s_t into peaky extremes.')
parser.add_argument('--w-dice', default=1.0, type=float,
                    help='Primary segmentation objective (region IoU). Aligned with '
                         'binary mAP@IoU evaluation metric.')
parser.add_argument('--w-margin', default=0.5, type=float,
                    help='Insurance for valley between event and background '
                         '(largely redundant with Dice but useful in early epochs).')
parser.add_argument('--w-var', default=0.1, type=float,
                    help='Plateau / anti-spike regulariser within events.')
parser.add_argument('--margin-m', default=0.3, type=float,
                    help='Valley margin in margin_loss.')
parser.add_argument('--margin-temp', default=5.0, type=float,
                    help='Temperature for soft-min/max in margin_loss. Lower '
                         'spreads gradient over more frames; higher approximates '
                         'true min/max but concentrates gradient on extremes.')

# Ablation extras (default 0 = inactive). Flip one of these to 1.0 with
# --w-{bce,dice,margin,var} 0 to run a single-loss ablation.
parser.add_argument('--w-focal', default=0.0, type=float,
                    help='Focal BCE weight. Down-weights easy frames via (1-p)^gamma.')
parser.add_argument('--focal-gamma', default=2.0, type=float)
parser.add_argument('--w-tversky', default=0.0, type=float,
                    help='Tversky loss weight. Penalises FN more than FP when beta>alpha.')
parser.add_argument('--tversky-alpha', default=0.3, type=float,
                    help='FP penalty in Tversky.')
parser.add_argument('--tversky-beta', default=0.7, type=float,
                    help='FN penalty in Tversky. beta>alpha forces event coverage.')
parser.add_argument('--w-hinge', default=0.0, type=float,
                    help='Per-frame hinge loss weight. Directly enforces P1+P2 '
                         '(event frames > threshold, non-event < threshold).')
parser.add_argument('--hinge-threshold', default=0.5, type=float,
                    help='Decision threshold for hinge_coverage_loss. '
                         'Should match the segment-extraction threshold used at eval.')

# Paths
parser.add_argument('--model-path', default='final_model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='final_model/checkpoint.pth')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')
parser.add_argument('--hivau-json-path', default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
parser.add_argument('--log-dir', default='logs')

# Eval
parser.add_argument('--score-source', default='dbranch',
                    choices=['dbranch', 'prob1', 'prob2'],
                    help='Which head to read for AUC/AP/mAP. Default dbranch (s_t). '
                         'Use prob1 to reproduce stock VadCLIP baseline.')
