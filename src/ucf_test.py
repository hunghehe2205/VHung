import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from intern_vad import VadInternVL
from src.utils.dataset import UCFDataset
from src.utils.tools import get_batch_mask
import src.ucf_option as ucf_option


def test(model, testdataloader, maxlen, gt, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
            # item[3] = frame_gt (unused in test)

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            num_splits = int(len_cur / maxlen) + 1
            lengths = torch.zeros(num_splits)
            remaining = len_cur
            for j in range(num_splits):
                if remaining >= maxlen:
                    lengths[j] = maxlen
                    remaining -= maxlen
                else:
                    lengths[j] = remaining
            lengths = lengths.to(int)

            logits = model(visual, lengths)
            logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
            prob = torch.sigmoid(logits[0:len_cur].squeeze(-1))

            if i == 0:
                ap = prob
            else:
                ap = torch.cat([ap, prob], dim=0)

    ap = ap.cpu().numpy().tolist()

    AUC = roc_auc_score(gt, np.repeat(ap, 16))
    AP = average_precision_score(gt, np.repeat(ap, 16))

    print("AUC: ", AUC, " AP: ", AP)

    return AUC, AP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    gt = np.load(args.gt_path)

    model = VadInternVL(
        args.visual_length, args.visual_width, args.visual_head,
        args.visual_layers, args.attn_window, device
    )
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length, gt, device)
