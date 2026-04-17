"""Render side-by-side markdown comparison of two diagnostic JSONs.

Usage:
  python src/diag_compare.py --a docs/diag/baseline.json \
                              --b docs/diag/exp12.json \
                              --out docs/diag/baseline_vs_exp12.md
"""
import argparse
import json
import os
import numpy as np


def _pct(vs, pred):
    if not vs:
        return 0.0
    return 100.0 * sum(1 for v in vs if pred(v)) / len(vs)


def _median(vs):
    return float(np.median(vs)) if vs else 0.0


def _row(label, a, b, fmt='{:.3f}'):
    try:
        d = fmt.format(b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ''
    except Exception:
        d = ''
    a_str = fmt.format(a) if isinstance(a, (int, float)) else str(a)
    b_str = fmt.format(b) if isinstance(b, (int, float)) else str(b)
    return f'| {label} | {a_str} | {b_str} | {d} |'


def _section(title):
    return f'\n## {title}\n\n| Metric | A | B | Δ (B − A) |\n|---|---:|---:|---:|'


def render(a, b, a_name, b_name):
    lines = [f'# Localization diagnostics comparison',
             '',
             f'- **A** = `{a.get("model_path", a_name)}`',
             f'- **B** = `{b.get("model_path", b_name)}`',
             f'- Test set: {a["n_videos"]["anomaly"]} anomaly + {a["n_videos"]["normal"]} normal videos',
             '',
             '## D1 — Score distribution (medians per video)',
             '',
             '| Metric | A | B | Δ |',
             '|---|---:|---:|---:|']

    for group in ['normal', 'anomaly']:
        lines.append(f'| **{group.upper()}** | | | |')
        for key, fmt in [('max', '{:.3f}'), ('mean', '{:.3f}'),
                         ('std', '{:.3f}'), ('median', '{:.3f}'),
                         ('n_peaks_abs_0.5', '{:.2f}'),
                         ('n_peaks_rel_0.8', '{:.2f}'),
                         ('n_peaks_p90',     '{:.2f}'),
                         ('peak_pos', '{:.3f}')]:
            av = _median(a['D1'][group][key])
            bv = _median(b['D1'][group][key])
            lines.append(_row(f'&nbsp;&nbsp;{key}', av, bv, fmt))

    lines.append('\n### Normal vs Anomaly separation (median max_prob)\n')
    a_sep = _median(a['D1']['anomaly']['max']) - _median(a['D1']['normal']['max'])
    b_sep = _median(b['D1']['anomaly']['max']) - _median(b['D1']['normal']['max'])
    lines.append('| Metric | A | B | Δ |')
    lines.append('|---|---:|---:|---:|')
    lines.append(_row('separation (anom_median − norm_median)', a_sep, b_sep))

    lines.append(_section('D3 — Peak location on anomaly videos'))
    lines.append(_row('peak_in_gt_ratio',     a['D3']['peak_in_gt_ratio'],
                                              b['D3']['peak_in_gt_ratio']))
    lines.append(_row('peak_in_middle_ratio', a['D3']['peak_in_middle_ratio'],
                                              b['D3']['peak_in_middle_ratio']))
    lines.append(_row('mean_peak_to_gt (frames)', a['D3']['mean_peak_to_gt'],
                                                  b['D3']['mean_peak_to_gt'], '{:.1f}'))

    lines.append(_section('D4 — IoU histogram (anomaly videos)'))
    lines.append(_row('median_iou',   a['D4']['median_iou'],   b['D4']['median_iou']))
    lines.append(_row('pct IoU ≥ 0.5', a['D4']['pct_iou_ge_0.5'] * 100,
                                        b['D4']['pct_iou_ge_0.5'] * 100, '{:.1f}%'))
    lines.append(_row('pct IoU < 0.1', a['D4']['pct_iou_lt_0.1'] * 100,
                                        b['D4']['pct_iou_lt_0.1'] * 100, '{:.1f}%'))

    lines.append('\n**IoU bin distribution (counts / % of anomaly videos)**\n')
    bins = a['D4']['iou_hist_bins']  # e.g. [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    hist_a = a['D4']['iou_hist_all']
    hist_b = b['D4']['iou_hist_all']
    bin_labels = []
    for i in range(len(hist_a)):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else 1.0
        bin_labels.append(f'[{lo:.1f}, {hi:.1f}{")" if i < len(hist_a) - 1 else "]"}')
    lines.append('| Bin | A count | A % | B count | B % |')
    lines.append('|---|---:|---:|---:|---:|')
    total_a = sum(hist_a) or 1
    total_b = sum(hist_b) or 1
    for i, lab in enumerate(bin_labels):
        ca = hist_a[i]
        cb = hist_b[i]
        lines.append(f'| {lab} | {ca} | {ca/total_a*100:.1f}% '
                     f'| {cb} | {cb/total_b*100:.1f}% |')

    lines.append(_section('D5 — Coverage quality (anomaly videos)'))
    for key in ['coverage_inside', 'spillover_ratio_window', 'spillover_ratio_global',
                'boundary_sharpness', 'over_coverage_ratio']:
        av = a['D5']['summary'][key]['median']
        bv = b['D5']['summary'][key]['median']
        lines.append(_row(f'{key} (median)', av, bv))

    lines.append('\n### D5 stratified by GT duration\n')
    lines.append('| Stratum | Metric | A | B | Δ |')
    lines.append('|---|---|---:|---:|---:|')
    for stratum in ['short', 'long']:
        for key, label in [('coverage_inside_median', 'coverage_inside'),
                           ('spillover_window_median', 'spillover_window'),
                           ('boundary_sharp_median',  'boundary_sharp'),
                           ('over_cov_median',        'over_cov')]:
            av = a['D5']['stratified'][stratum][key]
            bv = b['D5']['stratified'][stratum][key]
            lines.append(f'| {stratum} | {label} | {av:.3f} | {bv:.3f} | {bv-av:+.3f} |')

    lines.append('\n---\n')
    lines.append('## Interpretation hints\n')
    lines.append('- **coverage_inside ↑ but spillover_ratio still high** → next change: target spillover (asymmetric Dice / edge loss).')
    lines.append('- **spillover ↓ but coverage_inside low** → next change: inside-GT recall (CLAS2 top-k-from-GT).')
    lines.append('- **boundary_sharpness flat both sides** → current losses do not produce edges → revive/redesign boundary heads.')
    lines.append('- **over_coverage_ratio still > 2** → mean-contrast loophole confirmed → REPLACE with peak-contrast or Tversky.')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, help='Baseline JSON')
    parser.add_argument('--b', required=True, help='Target JSON (e.g. exp12)')
    parser.add_argument('--out', required=True, help='Output markdown path')
    args = parser.parse_args()

    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    md = render(a, b, os.path.basename(args.a), os.path.basename(args.b))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(md)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
