import argparse

parser = argparse.ArgumentParser(description='VadCLIP-UCF')
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

# Training
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[4, 8], nargs='+', type=int)

# Paths
parser.add_argument('--model-path', default='final_model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='final_model/checkpoint.pth')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

# logits3 anomaly map head
parser.add_argument('--hivau-json-path', default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
parser.add_argument('--lambda-bce', default=1.0, type=float)
parser.add_argument('--lambda-smooth', default=0.1, type=float)
parser.add_argument('--lambda-coverage', default=1.0, type=float)
parser.add_argument('--lambda-ranking', default=1.0, type=float)
parser.add_argument('--coverage-threshold', default=0.5, type=float)
parser.add_argument('--ranking-margin', default=0.3, type=float)
parser.add_argument('--log-dir', default='logs')
parser.add_argument('--smooth-sigma', default=1.5, type=float)

