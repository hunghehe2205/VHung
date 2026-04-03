import argparse

parser = argparse.ArgumentParser(description='InternVAD-UCF-v2')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=1024, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=16, type=int)

parser.add_argument('--max-epoch', default=15, type=int)
parser.add_argument('--model-path', default='model/model_ucf_intern_v2.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint_intern_v2.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_intern_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_intern_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')

parser.add_argument('--lr', default=5e-4)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[6, 9])
