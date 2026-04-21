import argparse

parser = argparse.ArgumentParser(description='VadCLIP-UCF Exp19 (TCN head hướng A1)')
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
parser.add_argument('--max-epoch', default=20, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Backbone lr (Transformer+GCN+classifier+text)')
parser.add_argument('--lr-tcn', default=1e-4, type=float,
                    help='TCN head lr (separate AdamW param group)')
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[6, 11], nargs='+', type=int)

# Paths
parser.add_argument('--model-path', default='final_model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='final_model/checkpoint.pth')
parser.add_argument('--load-baseline', default='model/model_ucf.pth',
                    help='Warm-start from this checkpoint (strict=False)')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')
parser.add_argument('--train-json',
                    default='HIVAU-70k-NEW/ucf_database_train_filtered.json')

# Curriculum
parser.add_argument('--phase1-epochs', default=3, type=int,
                    help='Epochs of CLAS2+CLASM warmup (TCN inactive)')
parser.add_argument('--phase2-epochs', default=6, type=int,
                    help='Up to this, only tcn_bce is active; after, dice+ctr join')

# Loss weights
parser.add_argument('--lambda-clas2', default=1.0, type=float,
                    help='Weight on CLAS2 (top-k MIL). 0 = ablate backbone anchor.')
parser.add_argument('--lambda-nce', default=1.0, type=float)
parser.add_argument('--lambda-cts', default=0.1, type=float,
                    help='Internal weight applied to loss_cts')
parser.add_argument('--contrast-margin', default=0.3, type=float,
                    help='Margin for tcn_ctr within-video contrast loss')

# TCN head
parser.add_argument('--tcn-pos-weight', default=6.0, type=float,
                    help='pos_weight scalar for tcn_bce (Gaussian target)')
parser.add_argument('--gauss-sigma', default=2.0, type=float,
                    help='Gaussian smoothing σ for TCN target (snippet units)')
parser.add_argument('--tcn-dilations', default=[1, 2, 4], nargs='+', type=int,
                    help='Dilations per 3x3 TCN conv block (RF scales with sum).')
parser.add_argument('--tcn-input', default='xpre',
                    choices=['xpre', 'concat_detach', 'concat_joint'],
                    help="TCN input path. 'xpre' = x_pre only (baseline); "
                         "'concat_detach' = cat(x_pre, visual_features.detach()) [F1]; "
                         "'concat_joint' = cat(x_pre, visual_features) [F2, joint grad].")
parser.add_argument('--lambda-tcn-dice', default=1.0, type=float,
                    help='Multiplier on tcn_dice during P3 (0 = ablate).')
parser.add_argument('--lambda-tcn-ctr', default=1.0, type=float,
                    help='Multiplier on tcn_ctr during P3 (0 = ablate).')

# Eval
parser.add_argument('--eval-head', default='tcn', choices=['tcn', 'wsv'],
                    help="Primary head for localization metrics")
