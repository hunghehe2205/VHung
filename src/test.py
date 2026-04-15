import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.detection_map import getDetectionMAP as dmAP
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device,
         quiet=False):
    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        ap1_per_video = []
        iterator = testdataloader if quiet else tqdm(
            testdataloader, desc='Testing', disable=not sys.stderr.isatty())
        for i, item in enumerate(iterator):
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

            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            ap1_per_video.append(prob1.cpu().numpy())

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    from utils.detection_map import getDetectionMAP_agnostic

    # Per-class (legacy)
    dmap_pc, iou = dmAP(element_logits2_stack, gtsegments, gtlabels,
                       excludeNormal=False)
    averageMAP_pc = float(np.mean(dmap_pc[:5]))

    # Class-agnostic: upsample each per-video prob1 array ×16 to frame granularity
    agnostic_stack = [np.repeat(fs, 16) for fs in ap1_per_video]
    dmap_ag, _ = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
    averageMAP_ag = float(np.mean(dmap_ag))

    if not quiet:
        print(f"AUC1={ROC1:.4f} AP1={AP1:.4f} | AUC2={ROC2:.4f} AP2={AP2:.4f}")
        pc_str = '/'.join(f'{v:.2f}' for v in dmap_pc[:5])
        ag_str = '/'.join(f'{v:.2f}' for v in dmap_ag[:5])
        print(f"[per-class] AVG={averageMAP_pc:.2f} [{pc_str}]")
        print(f"[agnostic ] AVG={averageMAP_ag:.2f} [{ag_str}]")

    return ROC1, averageMAP_ag, dmap_ag


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False, map_location=device))

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
