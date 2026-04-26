#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def mean_std(vals):
    vals = [float(x) for x in vals]
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return m, s


def load_json(p: Path):
    return json.loads(p.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description='Build final Person B report from selected/eval artifacts')
    parser.add_argument('--prefix', default='personb_tabred_final')
    parser.add_argument('--convention-share-training-batches', action='store_true')
    args = parser.parse_args()

    root = Path('.')
    results = root / 'results'
    sel = load_json(results / f'{args.prefix}_selected_runs.json')

    if sel.get('missing_expected_runs'):
        raise SystemExit('Cannot build final report: missing expected runs in grid')

    selected_rows = sel['selected_by_val']

    csv_path = results / f'{args.prefix}_summary.csv'
    fields = [
        'dataset', 'seed',
        'baseline_run', 'baseline_test',
        'selected_lambda', 'selected_run', 'selected_test', 'test_delta',
        'baseline_clean_score', 'selected_clean_score',
        'baseline_ensemble_gain', 'selected_ensemble_gain',
        'baseline_ece', 'selected_ece',
        'baseline_nll', 'selected_nll',
        'baseline_brier', 'selected_brier',
        'baseline_mask_mild_delta', 'selected_mask_mild_delta',
        'baseline_mask_moderate_delta', 'selected_mask_moderate_delta',
        'baseline_noise_mild_delta', 'selected_noise_mild_delta',
        'baseline_noise_moderate_delta', 'selected_noise_moderate_delta',
        'share_training_batches',
    ]

    out_rows = []
    by_dataset = {}

    for r in selected_rows:
        ds = r['dataset']
        sd = r['seed']
        b_eval = results / f'{args.prefix}_eval' / ds / f'seed{sd}' / 'baseline'
        s_eval = results / f'{args.prefix}_eval' / ds / f'seed{sd}' / 'selected'

        b_clean = load_json(b_eval / 'clean.json')
        s_clean = load_json(s_eval / 'clean.json')

        def corr_delta(eval_dir: Path, name: str):
            c = load_json(eval_dir / 'clean.json')['metrics']['score']
            x = load_json(eval_dir / f'{name}.json')['metrics']['score']
            return float(x - c)

        share_batches = bool(r.get('share_training_batches', True))

        row = {
            'dataset': ds,
            'seed': sd,
            'baseline_run': r['baseline_run'],
            'baseline_test': float(r['baseline_test']),
            'selected_lambda': float(r['selected_lambda']),
            'selected_run': r['selected_run'],
            'selected_test': float(r['selected_test']),
            'test_delta': float(r['test_delta']),
            'baseline_clean_score': float(b_clean['metrics']['score']),
            'selected_clean_score': float(s_clean['metrics']['score']),
            'baseline_ensemble_gain': float(b_clean['ensemble_gain']),
            'selected_ensemble_gain': float(s_clean['ensemble_gain']),
            'baseline_ece': float(b_clean['calibration']['ece']),
            'selected_ece': float(s_clean['calibration']['ece']),
            'baseline_nll': float(b_clean['calibration']['nll']),
            'selected_nll': float(s_clean['calibration']['nll']),
            'baseline_brier': float(b_clean['calibration']['brier']),
            'selected_brier': float(s_clean['calibration']['brier']),
            'baseline_mask_mild_delta': corr_delta(b_eval, 'mask_mild'),
            'selected_mask_mild_delta': corr_delta(s_eval, 'mask_mild'),
            'baseline_mask_moderate_delta': corr_delta(b_eval, 'mask_moderate'),
            'selected_mask_moderate_delta': corr_delta(s_eval, 'mask_moderate'),
            'baseline_noise_mild_delta': corr_delta(b_eval, 'noise_mild'),
            'selected_noise_mild_delta': corr_delta(s_eval, 'noise_mild'),
            'baseline_noise_moderate_delta': corr_delta(b_eval, 'noise_moderate'),
            'selected_noise_moderate_delta': corr_delta(s_eval, 'noise_moderate'),
            'share_training_batches': share_batches,
        }
        out_rows.append(row)
        by_dataset.setdefault(ds, []).append(row)

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    all_share_true = all(bool(r['share_training_batches']) for r in out_rows)

    md = []
    md.append('# Person B Final Report (TabReD-ready Pipeline)')
    md.append('')
    md.append('## Protocol')
    md.append('')
    md.append(f"- Run root: `{sel['protocol']['run_root']}`")
    md.append(f"- Datasets: {', '.join(sel['protocol']['datasets'])}")
    md.append(f"- Seeds: {sel['protocol']['seeds']}")
    md.append(f"- Lambda grid: {sel['protocol']['lambdas']}")
    md.append(f"- NCL space: `{sel['protocol']['ncl_space']}`")
    md.append(f"- Selection rule: validation score only (tie-break: `{sel['protocol']['tie_break']}`)")
    md.append(f"- Team convention `share_training_batches=true` required: {sel['protocol']['require_share_training_batches']}")
    md.append(f"- Team convention observed in selected runs: {all_share_true}")
    md.append('')

    md.append('## Final Test (Mean ± Std over seeds)')
    md.append('')
    md.append('| Dataset | Baseline | Validation-selected NCL | Delta |')
    md.append('|---|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        bm, bs = mean_std([r['baseline_test'] for r in rows])
        nm, ns = mean_std([r['selected_test'] for r in rows])
        md.append(f'| {ds} | {bm:.6f} ± {bs:.6f} | {nm:.6f} ± {ns:.6f} | {nm-bm:+.6f} |')

    md.append('')
    md.append('## Selected Lambda Distribution (by validation)')
    md.append('')
    md.append('| Dataset | Seed 0 | Seed 1 | Seed 2 |')
    md.append('|---|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        pick = {r['seed']: r['selected_lambda'] for r in rows}
        md.append(f"| {ds} | {pick.get(0, float('nan')):.4g} | {pick.get(1, float('nan')):.4g} | {pick.get(2, float('nan')):.4g} |")

    md.append('')
    md.append('## Diversity and Ensemble Gain (Clean Test, Mean over seeds)')
    md.append('')
    md.append('| Dataset | Ensemble gain baseline | Ensemble gain selected | Delta |')
    md.append('|---|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        bm, _ = mean_std([r['baseline_ensemble_gain'] for r in rows])
        sm, _ = mean_std([r['selected_ensemble_gain'] for r in rows])
        md.append(f'| {ds} | {bm:.6f} | {sm:.6f} | {sm-bm:+.6f} |')

    md.append('')
    md.append('## Robustness Mini-Ablation (Score delta vs clean, mean over seeds)')
    md.append('')
    md.append('| Dataset | Model | Mask mild | Mask moderate | Noise mild | Noise moderate |')
    md.append('|---|---|---:|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        for model in ['baseline', 'selected']:
            mmu, _ = mean_std([r[f'{model}_mask_mild_delta'] for r in rows])
            mou, _ = mean_std([r[f'{model}_mask_moderate_delta'] for r in rows])
            nmu, _ = mean_std([r[f'{model}_noise_mild_delta'] for r in rows])
            nou, _ = mean_std([r[f'{model}_noise_moderate_delta'] for r in rows])
            md.append(f'| {ds} | {model} | {mmu:.6f} | {mou:.6f} | {nmu:.6f} | {nou:.6f} |')

    report_path = results / f'{args.prefix}_report.md'
    report_path.write_text('\n'.join(md) + '\n')

    manifest = []
    manifest.append('# Person B Final Manifest (TabReD-ready)')
    manifest.append('')
    manifest.append('## Convention')
    manifest.append('')
    manifest.append('- share_training_batches=true (team-wide final convention)')
    manifest.append('- 3 fixed seeds per config')
    manifest.append('- validation-selected lambda only')
    manifest.append('')
    manifest.append('## Scripts')
    manifest.append('')
    manifest.append('- personb/check_tabred_ready.py')
    manifest.append('- personb/run_personb_tabred_final.sh')
    manifest.append('- personb/personb_tabred_final.pbs')
    manifest.append('- personb/prepare_selection_final.py')
    manifest.append('- personb/run_selected_eval_final.sh')
    manifest.append(f'- personb/build_final_report_final.py --prefix {args.prefix}')
    manifest.append('')
    manifest.append('## Outputs')
    manifest.append('')
    manifest.append(f'- {sel["protocol"]["run_root"]}')
    manifest.append(f'- results/{args.prefix}_selected_runs.json')
    manifest.append(f'- results/{args.prefix}_eval/')
    manifest.append(f'- results/{args.prefix}_summary.csv')
    manifest.append(f'- results/{args.prefix}_report.md')

    manifest_path = results / f'{args.prefix}_manifest.md'
    manifest_path.write_text('\n'.join(manifest) + '\n')

    print('wrote', csv_path)
    print('wrote', report_path)
    print('wrote', manifest_path)


if __name__ == '__main__':
    main()
