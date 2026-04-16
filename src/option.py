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

# dev_bce — supervised losses
parser.add_argument('--train-json',
                    default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
parser.add_argument('--pos-weight-path', default='list/pos_weight_bin.npy')
parser.add_argument('--pos-weight', default=None, type=float,
                    help='Override pos_weight scalar directly (skips --pos-weight-path)')
parser.add_argument('--phase1-epochs', default=3, type=int)
parser.add_argument('--phase2-epochs', default=6, type=int)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.1, type=float)
parser.add_argument('--focal-gamma', default=2.0, type=float,
                    help='0 = plain BCE, >0 = focal BCE with that gamma')
parser.add_argument('--phase3-loss', default='tv', choices=['tv', 'dice'],
                    help='Phase 3 localization loss: tv or dice')
parser.add_argument('--lambda-contrast', default=0.0, type=float,
                    help='Weight for within-video contrast loss (Phase 3). 0 = off')
parser.add_argument('--contrast-margin', default=0.3, type=float,
                    help='Margin for within-video contrast loss')
parser.add_argument('--lambda-boundary', default=0.5, type=float,
                    help='Weight for start/end boundary BCE (Phase 2+). 0 = off')
parser.add_argument('--boundary-sigma', default=1.0, type=float,
                    help='Gaussian sigma for boundary target smoothing (snippets)')
parser.add_argument('--boundary-pos-weight', default=10.0, type=float,
                    help='pos_weight for boundary BCE (positives are rare)')
parser.add_argument('--inference', default='threshold',
                    choices=['threshold', 'bsn'],
                    help='Proposal generation: adaptive threshold or BSN start/end')
parser.add_argument('--bsn-start-thresh', default=0.5, type=float,
                    help='BSN: relative peak threshold for start probs')
parser.add_argument('--bsn-end-thresh', default=0.5, type=float)
parser.add_argument('--bsn-max-dur', default=2048, type=int,
                    help='BSN: max proposal duration in frames (after upsample)')
parser.add_argument('--upsample', default='repeat', choices=['repeat', 'linear'],
                    help='Eval upsample mode: step-function (repeat) or linear interp')
parser.add_argument('--normalize', action='store_true',
                    help='Viz: min-max normalize prob per video to see shape')
