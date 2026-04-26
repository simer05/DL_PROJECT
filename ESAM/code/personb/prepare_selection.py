#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


def mean_std(xs):
    if not xs:
        return float('nan'), float('nan')
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v**0.5


def main() -> None:
    root = Path('outputs/personb_ncl_rigorous')
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)

    if not root.exists():
        raise SystemExit('Missing outputs/personb_ncl_rigorous')

    rows = []
    for report in sorted(root.glob('*/seed*/lambda_*/report.json')):
        if '-best-head/' in str(report) or '-greedy-heads/' in str(report):
            continue
        data = json.loads(report.read_text())
        cfg = data['config']
        ds = cfg['data']['path'].split('/')[-1]
        seed = int(cfg['seed'])
        lam = float(cfg.get('lambda_ncl', 0.0))
        warm = int(cfg.get('ncl_warmup_epochs', 0))
        space = cfg.get('ncl_space', 'logits')
        row = {
            'dataset': ds,
            'seed': seed,
            'lambda_ncl': lam,
            'warmup': warm,
            'space': space,
            'val_score': float(data['metrics']['val']['score']),
            'test_score': float(data['metrics']['test']['score']),
            'run_dir': str(report.parent),
            'config_path': str(report.parent / 'config.generated.toml'),
            'report_path': str(report),
        }
        rows.append(row)

    datasets = ['adult', 'california', 'covtype2']
    seeds = [0, 1, 2]
    lambdas = [0.0, 5e-4, 1e-3, 2e-3]

    filtered = [r for r in rows if r['dataset'] in datasets and r['seed'] in seeds and r['warmup'] == 10 and r['space'] == 'logits' and abs(r['lambda_ncl']) in [abs(x) for x in lambdas]]

    selected = []
    missing = []
    for ds in datasets:
        for sd in seeds:
            cand = [r for r in filtered if r['dataset'] == ds and r['seed'] == sd]
            have = sorted({r['lambda_ncl'] for r in cand})
            for lam in lambdas:
                if not any(abs(x - lam) < 1e-12 for x in have):
                    missing.append((ds, sd, lam))
            if not cand:
                continue
            best = max(cand, key=lambda x: x['val_score'])
            base = [r for r in cand if abs(r['lambda_ncl']) < 1e-12]
            if not base:
                missing.append((ds, sd, 0.0))
                continue
            base = base[0]
            selected.append({
                'dataset': ds,
                'seed': sd,
                'baseline_run': base['run_dir'],
                'baseline_val': base['val_score'],
                'baseline_test': base['test_score'],
                'selected_lambda': best['lambda_ncl'],
                'selected_run': best['run_dir'],
                'selected_val': best['val_score'],
                'selected_test': best['test_score'],
                'test_delta': best['test_score'] - base['test_score'],
            })

    out = {
        'datasets': datasets,
        'seeds': seeds,
        'lambdas': lambdas,
        'missing_expected_runs': [{'dataset': d, 'seed': s, 'lambda_ncl': l} for d, s, l in missing],
        'selected_by_val': selected,
    }
    (out_dir / 'personb_selected_runs.json').write_text(json.dumps(out, indent=2))

    # quick aggregate snapshot
    agg = {}
    for ds in datasets:
        ss = [x for x in selected if x['dataset'] == ds]
        if not ss:
            continue
        b = [x['baseline_test'] for x in ss]
        n = [x['selected_test'] for x in ss]
        bm, bs = mean_std(b)
        nm, ns = mean_std(n)
        agg[ds] = {
            'baseline_test_mean': bm,
            'baseline_test_std': bs,
            'selected_test_mean': nm,
            'selected_test_std': ns,
            'delta_mean': nm - bm,
        }
    (out_dir / 'personb_selected_snapshot.json').write_text(json.dumps(agg, indent=2))

    print('wrote', out_dir / 'personb_selected_runs.json')
    print('wrote', out_dir / 'personb_selected_snapshot.json')
    if missing:
        print('missing runs:', len(missing))
        for m in missing[:20]:
            print(' ', m)


if __name__ == '__main__':
    main()
