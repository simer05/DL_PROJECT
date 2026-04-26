#!/usr/bin/env python3
from __future__ import annotations

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


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return 'nan'
        return f'{x:.6f}'
    return str(x)


def main() -> None:
    root = Path('.')
    results = root / 'results'
    sel = load_json(results / 'personb_selected_runs.json')

    if sel.get('missing_expected_runs'):
        raise SystemExit('Cannot build final report: missing expected runs in rigorous grid.')

    selected_rows = sel['selected_by_val']

    # Build per-seed compact csv
    csv_path = results / 'personb_ncl_final_summary.csv'
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
    ]

    out_rows = []

    # Aggregation containers
    by_dataset = {}

    for r in selected_rows:
        ds = r['dataset']
        sd = r['seed']

        b_eval = results / 'personb_eval' / ds / f'seed{sd}' / 'baseline'
        s_eval = results / 'personb_eval' / ds / f'seed{sd}' / 'selected'

        b_clean = load_json(b_eval / 'clean.json')
        s_clean = load_json(s_eval / 'clean.json')

        def corr_delta(eval_dir: Path, name: str):
            c = load_json(eval_dir / 'clean.json')['metrics']['score']
            x = load_json(eval_dir / f'{name}.json')['metrics']['score']
            return float(x - c)

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
        }
        out_rows.append(row)
        by_dataset.setdefault(ds, []).append(row)

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # Warmup ablation (adult seed2)
    warmup_rows = []
    for tag in ['warmup0', 'warmup20']:
        rp = root / 'outputs' / 'personb_ncl_rigorous' / 'adult' / 'seed2' / f'lambda_0p001_{tag}' / 'report.json'
        if rp.exists():
            d = load_json(rp)
            warmup_rows.append((tag, float(d['metrics']['val']['score']), float(d['metrics']['test']['score'])))

    # Build markdown report
    md = []
    md.append('# Person B Final Report: TabM + NCL (Rigorous Protocol)')
    md.append('')
    md.append('## Protocol')
    md.append('')
    md.append('- Datasets: adult, california, covtype2.')
    md.append('- Seeds: 0, 1, 2.')
    md.append('- Lambda grid: 0, 5e-4, 1e-3, 2e-3.')
    md.append('- Selection rule: per-dataset, per-seed best lambda chosen by validation score only.')
    md.append('- Final test reported for selected run; test not used for lambda selection.')
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
        b = [r['baseline_ensemble_gain'] for r in rows]
        s = [r['selected_ensemble_gain'] for r in rows]
        bm, _ = mean_std(b)
        sm, _ = mean_std(s)
        md.append(f'| {ds} | {bm:.6f} | {sm:.6f} | {sm-bm:+.6f} |')

    md.append('')
    md.append('## Calibration (Classification datasets only, clean test mean over seeds)')
    md.append('')
    md.append('| Dataset | ECE baseline | ECE selected | NLL baseline | NLL selected | Brier baseline | Brier selected |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        if ds == 'california':
            continue
        be, _ = mean_std([r['baseline_ece'] for r in rows])
        se, _ = mean_std([r['selected_ece'] for r in rows])
        bn, _ = mean_std([r['baseline_nll'] for r in rows])
        sn, _ = mean_std([r['selected_nll'] for r in rows])
        bb, _ = mean_std([r['baseline_brier'] for r in rows])
        sb, _ = mean_std([r['selected_brier'] for r in rows])
        md.append(f'| {ds} | {be:.6f} | {se:.6f} | {bn:.6f} | {sn:.6f} | {bb:.6f} | {sb:.6f} |')

    md.append('')
    md.append('## Robustness Mini-Ablation (Inference-time corruption, score delta vs clean, mean over seeds)')
    md.append('')
    md.append('| Dataset | Model | Mask mild | Mask moderate | Noise mild | Noise moderate |')
    md.append('|---|---|---:|---:|---:|---:|')
    for ds, rows in sorted(by_dataset.items()):
        for model in ['baseline', 'selected']:
            mm = [r[f'{model}_mask_mild_delta'] for r in rows]
            mo = [r[f'{model}_mask_moderate_delta'] for r in rows]
            nm = [r[f'{model}_noise_mild_delta'] for r in rows]
            no = [r[f'{model}_noise_moderate_delta'] for r in rows]
            mmu, _ = mean_std(mm)
            mou, _ = mean_std(mo)
            nmu, _ = mean_std(nm)
            nou, _ = mean_std(no)
            md.append(f'| {ds} | {model} | {mmu:.6f} | {mou:.6f} | {nmu:.6f} | {nou:.6f} |')

    md.append('')
    md.append('## Additional Method Ablation: NCL Warmup (adult, seed2, lambda=1e-3)')
    md.append('')
    if warmup_rows:
        md.append('| Setting | Val score | Test score |')
        md.append('|---|---:|---:|')
        for tag, v, t in warmup_rows:
            md.append(f'| {tag} | {v:.6f} | {t:.6f} |')
    else:
        md.append('- Warmup ablation outputs were not found.')

    md.append('')
    md.append('## Failure Log and Recovery')
    md.append('')
    md.append('- Scheduler queue access denial for direct `g1` submission was handled by submitting via `normal` queue (routed to `g1`).')
    md.append('- Previous PyTorch 2.6 checkpoint loading issue was fixed by setting `weights_only=False` in `lib/util.py`.')
    md.append('')

    md.append('## Takeaway for Report Writing')
    md.append('')
    md.append('- Did NCL help: yes on adult, mixed/near-neutral on california and covtype2.')
    md.append('- Consistency: partial, depends on dataset/seed.')
    md.append('- Diversity increase: generally improved on classification settings where NCL was selected.')
    md.append('- Calibration/robustness: use the tables above; claim improvements only where deltas are consistently favorable.')
    md.append('- Safe final claim: explicit NCL regularization can improve TabM on some tabular tasks, but effects are dataset-dependent and require validation-based tuning.')

    report_path = results / 'personb_ncl_final_report.md'
    report_path.write_text('\n'.join(md) + '\n')

    manifest = []
    manifest.append('# Person B Experiment Manifest')
    manifest.append('')
    manifest.append('## Scripts Used')
    manifest.append('')
    manifest.append('- personb/run_experiment.py')
    manifest.append('- personb/run_personb_rigorous.sh')
    manifest.append('- personb/personb_ncl_rigorous.pbs')
    manifest.append('- personb/prepare_selection.py')
    manifest.append('- personb/evaluate_checkpoint.py')
    manifest.append('- personb/run_selected_eval.sh')
    manifest.append('- personb/build_final_report.py')
    manifest.append('')
    manifest.append('## Output Directories')
    manifest.append('')
    manifest.append('- outputs/personb_ncl_rigorous/')
    manifest.append('- results/personb_eval/')
    manifest.append('- results/personb_selected_runs.json')
    manifest.append('- results/personb_ncl_final_summary.csv')
    manifest.append('- results/personb_ncl_final_report.md')
    manifest.append('')
    manifest.append('## Commands')
    manifest.append('')
    manifest.append('```bash')
    manifest.append('qsub personb/personb_ncl_rigorous.pbs')
    manifest.append('bash personb/run_selected_eval.sh')
    manifest.append('python personb/build_final_report.py')
    manifest.append('```')

    (results / 'personb_experiment_manifest.md').write_text('\n'.join(manifest) + '\n')

    print('wrote', csv_path)
    print('wrote', report_path)
    print('wrote', results / 'personb_experiment_manifest.md')


if __name__ == '__main__':
    main()
