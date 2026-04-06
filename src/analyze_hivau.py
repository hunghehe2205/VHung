import json
import numpy as np
from collections import defaultdict

HIVAU_JSON = '/Users/hunghehe2205/Projects/VHung/HIVAU-70k-NEW/ucf_database_train.json'
CLIP_LEN = 16

with open(HIVAU_JSON) as f:
    data = json.load(f)

ratios = []
ratios_by_label = defaultdict(list)

for video_name, info in data.items():
    fps = info['fps']
    n_frames = info['n_frames']
    events = info['events']
    total_duration = n_frames / fps

    anomaly_duration = 0
    for start, end in events:
        anomaly_duration += (end - start)

    ratio = anomaly_duration / total_duration if total_duration > 0 else 0
    ratios.append(ratio)

    for label in info['label']:
        ratios_by_label[label].append(ratio)

ratios = np.array(ratios)

print("=" * 60)
print("HIVAU Anomaly Frame Ratio Analysis")
print("=" * 60)
print(f"Total anomaly videos: {len(ratios)}")
print(f"Mean anomaly ratio:   {ratios.mean():.4f} ({ratios.mean()*100:.1f}%)")
print(f"Median anomaly ratio: {np.median(ratios):.4f} ({np.median(ratios)*100:.1f}%)")
print(f"Std:                  {ratios.std():.4f}")
print(f"Min:                  {ratios.min():.4f} ({ratios.min()*100:.1f}%)")
print(f"Max:                  {ratios.max():.4f} ({ratios.max()*100:.1f}%)")

print(f"\nDistribution:")
for threshold in [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.0]:
    count = (ratios <= threshold).sum()
    print(f"  <= {threshold*100:5.1f}%: {count:4d} videos ({count/len(ratios)*100:.1f}%)")

print(f"\n{'Label':<16} {'Count':>5} {'Mean%':>7} {'Median%':>8} {'Min%':>6} {'Max%':>6}")
print("-" * 60)
for label in sorted(ratios_by_label.keys()):
    r = np.array(ratios_by_label[label])
    print(f"{label:<16} {len(r):>5} {r.mean()*100:>6.1f}% {np.median(r)*100:>7.1f}% {r.min()*100:>5.1f}% {r.max()*100:>5.1f}%")
