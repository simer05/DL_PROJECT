#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _parse_runtime_minutes(v: str | None) -> float | None:
    if not v:
        return None
    try:
        h, m, s = v.split(':')
        return int(h) * 60.0 + int(m) + float(s) / 60.0
    except Exception:
        return None


def _discover_runs(run_root: Path) -> list[dict]:
    rows: list[dict] = []
    for report in run_root.glob('**/report.json'):
        out_dir = report.parent
        run_meta_path = out_dir / 'run_meta.json'
        cfg_path = out_dir / 'config.generated.toml'

        run_meta = {}
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text())

        rel = report.relative_to(run_root)
        parts = rel.parts
        dataset = parts[0] if len(parts) > 0 else run_meta.get('dataset', 'unknown')
        seed_tag = next((p for p in parts if p.startswith('seed')), f"seed{run_meta.get('seed', -1)}")
        trainfrac_tag = next((p for p in parts if p.startswith('trainfrac_')), None)
        seed = int(str(seed_tag).replace('seed', ''))
        train_fraction = run_meta.get('train_fraction')
        if train_fraction is None and trainfrac_tag:
            train_fraction = float(trainfrac_tag.replace('trainfrac_', '').replace('p', '.'))

        variant = run_meta.get('run_tag') or out_dir.name
        rho = run_meta.get('rho')
        if rho is None:
            rho_part = next((p for p in parts if p.startswith('rho_')), 'rho_0p0')
            rho = float(rho_part.replace('rho_', '').replace('m', '-').replace('p', '.'))

        r = json.loads(report.read_text())
        val = float((r.get('metrics', {}).get('val', {}) or {}).get('score'))
        test = float((r.get('metrics', {}).get('test', {}) or {}).get('score'))
        runtime_min = _parse_runtime_minutes(r.get('time'))

        esam_cfg = (r.get('config', {}) or {})
        use_esam = bool(esam_cfg.get('use_esam', rho > 0.0))
        esam_end_epoch = int(esam_cfg.get('esam_end_epoch', -1))
        schedule = 'full'
        if esam_end_epoch >= 0:
            schedule = f'stop@{esam_end_epoch}'

        rows.append(
            {
                'dataset': dataset,
                'train_fraction': float(train_fraction),
                'seed': seed,
                'variant': str(variant),
                'use_esam': use_esam,
                'rho': float(rho),
                'schedule': schedule,
                'val_score': val,
                'test_score': test,
                'runtime_minutes': runtime_min,
                'checkpoint_path': str(out_dir),
                'report_path': str(report),
                'config_path': str(cfg_path),
                'log_path': '',
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--selection-margin', type=float, default=0.0002)
    args = p.parse_args()

    rows = _discover_runs(args.run_root)
    if not rows:
        raise SystemExit('No runs discovered under run-root')

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['train_fraction'], row['seed'])].append(row)

    selected: list[dict] = []
    for (dataset, train_fraction, seed), items in sorted(grouped.items()):
        baseline = next((x for x in items if (not x['use_esam']) or x['rho'] == 0.0 or x['variant'] == 'baseline'), None)
        if baseline is None:
            best = max(items, key=lambda x: x['val_score'])
            best = {**best, 'selected_by_validation': True, 'selected_reason': 'no_baseline_found'}
            selected.append(best)
            continue

        best = max(items, key=lambda x: x['val_score'])
        if best['variant'] == baseline['variant']:
            pick = {**baseline, 'selected_by_validation': True, 'selected_reason': 'baseline_best_val'}
        elif best['val_score'] >= baseline['val_score'] + args.selection_margin:
            pick = {**best, 'selected_by_validation': True, 'selected_reason': 'esam_margin_win'}
        else:
            pick = {**baseline, 'selected_by_validation': True, 'selected_reason': 'baseline_fallback_margin'}

        pick['baseline_val_score'] = baseline['val_score']
        pick['baseline_test_score'] = baseline['test_score']
        pick['test_delta_vs_baseline'] = pick['test_score'] - baseline['test_score']
        selected.append(pick)

    payload = {
        'protocol': {
            'selection': 'validation_only_with_margin',
            'selection_margin': float(args.selection_margin),
            'metric_direction': 'higher_is_better_score_field',
        },
        'all_rows': rows,
        'selected_rows': selected,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2))
    print(f'[done] {args.result} rows={len(rows)} selected={len(selected)}')


if __name__ == '__main__':
    main()
