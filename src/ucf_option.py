import argparse

parser = argparse.ArgumentParser(description='InternVAD-UCF')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--visual-length', default=512, type=int)
parser.add_argument('--visual-width', default=1024, type=int)
parser.add_argument('--visual-head', default=8, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf_intern_param.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint_intern_param.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_intern_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_intern_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')

parser.add_argument('--lr', default=1e-4)
parser.add_argument('--scheduler-rate', default=0.3)
parser.add_argument('--scheduler-milestones', default=[6, 9])

# Frame-level supervision (HIVAU)
parser.add_argument('--hivau-path', default='HIVAU-70k-NEW/ucf_database_train.json')
parser.add_argument('--lambda-frame', default=0.1, type=float)
parser.add_argument('--mu-smooth', default=0.5, type=float)
parser.add_argument('--normal-target', default=0.05, type=float)
parser.add_argument('--gauss-sigma', default=1.5, type=float)
