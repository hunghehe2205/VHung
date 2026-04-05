"""
Generate CSV lists and ground truth files for UCF-Crime dataset.

Usage:
    cd src/
    python make_list.py --feature-root /path/to/UCFClipFeatures

Outputs (in list/ folder):
    - ucf_CLIP_rgb.csv       (train list)
    - ucf_CLIP_rgbtest.csv   (test list)
    - gt_ucf.npy             (frame-level GT)
    - gt_segment_ucf.npy     (segment-level GT)
    - gt_label_ucf.npy       (segment labels)
"""
import argparse
import os
import csv
import numpy as np
import pandas as pd

CLIP_LEN = 16


def make_train_list(feature_root, train_txt, output_csv):
    files = [f.strip() for f in open(train_txt)]
    normal = []

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for file in files:
            # Anomaly_Train.txt: "Abuse/Abuse001_x264.mp4"
            # Feature path: UCFClipFeatures/Abuse/Abuse001_x264__0.npy
            label = file.split('/')[0]
            video_name = file.split('/')[-1].replace('.mp4', '')
            base_path = os.path.join(feature_root, label, video_name + '__0.npy')

            if not os.path.exists(base_path):
                print(f"[MISS] {base_path}")
                continue

            prefix = base_path[:-5]  # remove "0.npy"
            if 'Normal' in label:
                for i in range(10):
                    normal.append(prefix + str(i) + '.npy')
            else:
                for i in range(10):
                    writer.writerow([prefix + str(i) + '.npy', label])

        for file in normal:
            writer.writerow([file, 'Normal'])

    print(f"[OK] Train list: {output_csv}")


def make_test_list(feature_root, test_txt, output_csv):
    files = [f.strip() for f in open(test_txt)]
    normal = []

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for file in files:
            label = file.split('/')[0]
            video_name = file.split('/')[-1].replace('.mp4', '')
            base_path = os.path.join(feature_root, label, video_name + '__0.npy')

            if not os.path.exists(base_path):
                print(f"[MISS] {base_path}")
                continue

            prefix = base_path[:-5]
            if 'Normal' in label:
                for i in range(10):
                    normal.append(prefix + str(i) + '.npy')
            else:
                for i in range(10):
                    writer.writerow([prefix + str(i) + '.npy', label])

        for file in normal:
            writer.writerow([file, 'Normal'])

    print(f"[OK] Test list: {output_csv}")


def make_gt(test_csv, annotation_txt, output_path):
    gt_lines = [l.strip() for l in open(annotation_txt)]
    lists = pd.read_csv(test_csv)
    gt = []

    for idx in range(lists.shape[0]):
        name = lists.loc[idx]['path']
        if '__0.npy' not in name:
            continue

        fea = np.load(name)
        lens = (fea.shape[0] + 1) * CLIP_LEN
        video_name = os.path.basename(name)[:-7]  # remove "__0.npy"

        gt_vec = np.zeros(lens).astype(np.float32)
        if 'Normal' not in video_name:
            for gt_line in gt_lines:
                if video_name in gt_line:
                    gt_content = gt_line.split('  ')[1:-1]
                    abnormal_fragment = [
                        [int(gt_content[i]), int(gt_content[j])]
                        for i in range(1, len(gt_content), 2)
                        for j in range(2, len(gt_content), 2)
                        if j == i + 1
                    ]
                    if len(abnormal_fragment) != 0:
                        for frag in abnormal_fragment:
                            if frag[0] != -1 and frag[1] != -1:
                                gt_vec[frag[0]:frag[1]] = 1.0
                    break
        gt.extend(gt_vec[:-CLIP_LEN])

    np.save(output_path, gt)
    print(f"[OK] GT: {output_path}")


def make_gt_segment(test_csv, annotation_txt, segment_path, label_path):
    gt_lines = [l.strip() for l in open(annotation_txt)]
    lists = pd.read_csv(test_csv)
    gt_segment = []
    gt_label = []

    for idx in range(lists.shape[0]):
        name = lists.loc[idx]['path']
        label_text = lists.loc[idx]['label']
        if '__0.npy' not in name:
            continue

        segment = []
        label = []
        if 'Normal' in label_text:
            fea = np.load(name)
            lens = fea.shape[0] * CLIP_LEN
            segment.append([0, lens])
            label.append('A')
        else:
            video_name = os.path.basename(name)[:-7]
            for gt_line in gt_lines:
                if video_name in gt_line:
                    gt_content = gt_line.split('  ')
                    segment.append([gt_content[2], gt_content[3]])
                    label.append(gt_content[1])
                    if gt_content[4] != '-1':
                        segment.append([gt_content[4], gt_content[5]])
                        label.append(gt_content[1])
                    break
        gt_segment.append(segment)
        gt_label.append(label)

    np.save(segment_path, gt_segment)
    np.save(label_path, gt_label)
    print(f"[OK] GT segments: {segment_path}")
    print(f"[OK] GT labels: {label_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-root', required=True, help='Path to UCFClipFeatures/')
    parser.add_argument('--data-dir', default='../data', help='Path to data/ with txt annotations')
    parser.add_argument('--list-dir', default='list', help='Output directory for lists')
    args = parser.parse_args()

    os.makedirs(args.list_dir, exist_ok=True)

    train_txt = os.path.join(args.data_dir, 'Anomaly_Train.txt')
    test_txt = os.path.join(args.data_dir, 'Anomaly_Test.txt')
    annotation_txt = os.path.join(args.data_dir, 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt')

    train_csv = os.path.join(args.list_dir, 'ucf_CLIP_rgb.csv')
    test_csv = os.path.join(args.list_dir, 'ucf_CLIP_rgbtest.csv')
    gt_path = os.path.join(args.list_dir, 'gt_ucf.npy')
    gt_segment_path = os.path.join(args.list_dir, 'gt_segment_ucf.npy')
    gt_label_path = os.path.join(args.list_dir, 'gt_label_ucf.npy')

    # Step 1: Generate CSV lists
    make_train_list(args.feature_root, train_txt, train_csv)
    make_test_list(args.feature_root, test_txt, test_csv)

    # Step 2: Generate ground truth (requires test CSV + features)
    make_gt(test_csv, annotation_txt, gt_path)
    make_gt_segment(test_csv, annotation_txt, gt_segment_path, gt_label_path)

    print("\nDone! All list files generated.")
