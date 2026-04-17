"""Normal-video prediction diagnostic for CLIPVAD checkpoints.

Analyzes how the model scores Normal videos vs Anomaly videos:
- prob distribution (max, mean) per class
- % Normal with max_prob above {0.5, 0.7, 0.9}
- top-K worst Normal (highest max_prob)
- adaptive-threshold proposals per Normal video
- frame-level FP rate at absolute thresholds {0.3, 0.5, 0.7}
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask
import option


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _adaptive_thresh_proposals(prob, thr_ratio=0.6):
    """Returns list of (s, e, score) using `max - thr_ratio*(max-min)`."""
    if prob.max() == prob.min():
        return []
    thr = prob.max() - (prob.max() - prob.min()) * thr_ratio
    mask = np.concatenate([[0.0], (prob > thr).astype(np.float32), [0.0]])
    diff = mask[1:] - mask[:-1]
    starts = np.where(diff == 1)[0].tolist()
    ends = np.where(diff == -1)[0].tolist()
    out = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            out.append((s, e, float(prob[s:e].max())))
    return out


def _get_probs_per_video(model, loader, maxlen, prompt_text, device):
    """Returns (probs_list [np.array len_cur], labels_list [str], paths)."""
    probs, labels, paths = [], [], []
    df = loader.dataset.df
    with torch.no_grad():
        for i, item in enumerate(loader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, _, _, _ = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1],
                                      logits1.shape[2])
            prob = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
            probs.append(prob)
            labels.append(df.loc[i]['label'])
            paths.append(df.loc[i]['path'])
    return probs, labels, paths


def _dist(arr, label):
    arr = np.asarray(arr)
    if len(arr) == 0:
        print(f'  {label:20s}  n=0')
        return
    print(f'  {label:20s}  n={len(arr):3d}  '
          f'mean={arr.mean():.4f}  median={np.median(arr):.4f}  '
          f'p90={np.percentile(arr, 90):.4f}  max={arr.max():.4f}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    from utils.tools import get_prompt_text
    prompt_text = get_prompt_text(LABEL_MAP)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                     map_location=device), strict=False)
    model.to(device).eval()

    probs, labels, paths = _get_probs_per_video(
        model, testloader, args.visual_length, prompt_text, device)

    normal_idx = [i for i, l in enumerate(labels) if str(l).lower() == 'normal']
    anomaly_idx = [i for i, l in enumerate(labels) if str(l).lower() != 'normal']

    n_max = np.array([probs[i].max() for i in normal_idx])
    n_mean = np.array([probs[i].mean() for i in normal_idx])
    a_max = np.array([probs[i].max() for i in anomaly_idx])
    a_mean = np.array([probs[i].mean() for i in anomaly_idx])

    print('=' * 74)
    print(f'Model: {args.model_path}')
    print(f'Normal videos : {len(normal_idx)}   Anomaly videos: {len(anomaly_idx)}')
    print('=' * 74)

    print('\n[per-video max_prob distribution]')
    _dist(n_max, 'Normal max_prob')
    _dist(a_max, 'Anomaly max_prob')

    print('\n[per-video mean_prob distribution]')
    _dist(n_mean, 'Normal mean_prob')
    _dist(a_mean, 'Anomaly mean_prob')

    print('\n[% Normal videos with max_prob above threshold]')
    for thr in [0.3, 0.5, 0.7, 0.9]:
        n = int((n_max > thr).sum())
        print(f'  max_prob > {thr:.1f}   {n:3d}/{len(n_max)}  ({100.0*n/len(n_max):5.1f}%)')

    print('\n[top-10 worst Normal videos by max_prob]')
    order = np.argsort(-n_max)[:10]
    for rank, k in enumerate(order):
        idx = normal_idx[k]
        p = probs[idx]
        vid = os.path.basename(paths[idx]).replace('.npy', '')
        print(f'  {rank+1:2d}. {vid:40s}  max={p.max():.3f}  '
              f'mean={p.mean():.4f}  len={len(p)}')

    print('\n[adaptive-threshold proposals on Normal videos]')
    counts = []
    have_any = 0
    for k in normal_idx:
        props = _adaptive_thresh_proposals(probs[k], thr_ratio=0.6)
        counts.append(len(props))
        have_any += (len(props) > 0)
    counts = np.array(counts)
    print(f'  Normal videos with ≥1 proposal:  {have_any}/{len(normal_idx)}  '
          f'({100.0*have_any/len(normal_idx):.1f}%)')
    print(f'  Proposals/video          mean={counts.mean():.2f}  '
          f'median={int(np.median(counts))}  max={counts.max()}')

    print('\n[frame-level FP rate at absolute thresholds]')
    normal_frames = np.concatenate([probs[i] for i in normal_idx])
    anomaly_frames = np.concatenate([probs[i] for i in anomaly_idx])
    print(f'  Normal total frames : {len(normal_frames):6d}')
    print(f'  Anomaly total frames: {len(anomaly_frames):6d}')
    for thr in [0.3, 0.5, 0.7]:
        fpr = (normal_frames > thr).mean()
        tpr_anomaly = (anomaly_frames > thr).mean()
        print(f'  thr={thr:.1f}  FP-rate Normal={100.0*fpr:5.2f}%  '
              f'Anomaly-frames>thr={100.0*tpr_anomaly:5.2f}%')

    print('\n[class-agnostic summary]')
    print(f'  median(Normal max)  = {np.median(n_max):.3f}')
    print(f'  median(Anomaly max) = {np.median(a_max):.3f}')
    print(f'  separation (median Anomaly max - median Normal max) = '
          f'{np.median(a_max) - np.median(n_max):+.3f}')
    print('=' * 74)


if __name__ == '__main__':
    main()
