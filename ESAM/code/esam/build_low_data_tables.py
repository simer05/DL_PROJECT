#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _runtime_seconds(s: str | None) -> float | None:
    if not s:
        return None
    try:
        h, m, sec = s.split(':')
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None


def _mean_from_jsonl(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    vals = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        v = rec.get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _extract_losses(metrics_train: dict, metrics_val: dict) -> tuple[float | None, float | None]:
    for k in ['cross-entropy', 'rmse', 'mae']:
        if k in metrics_train and k in metrics_val:
            return float(metrics_train[k]), float(metrics_val[k])
    return None, None


def collect_rows(run_root: Path, diagnostics_root: Path) -> pd.DataFrame:
    rows = []
    for report_path in sorted(run_root.glob('**/report.json')):
        rel = report_path.relative_to(run_root)
        parts = rel.parts
        # dataset/trainfrac_x/seedY/start_Z/rho_x[/tag]/report.json
        if len(parts) < 6:
            continue
        dataset = parts[0]
        frac_tag = parts[1].replace('trainfrac_', '')
        seed = int(parts[2].replace('seed', ''))
        start_epoch = int(parts[3].replace('start_', ''))
        rho_tag = parts[4]
        rho = float(rho_tag.replace('rho_', '').replace('m', '-').replace('p', '.'))

        rep = json.loads(report_path.read_text())
        mtrain = rep['metrics']['train']
        mval = rep['metrics']['val']
        mtest = rep['metrics']['test']
        train_loss, val_loss = _extract_losses(mtrain, mval)
        train_val_gap = (val_loss - train_loss) if (train_loss is not None and val_loss is not None) else None

        diag_path = None
        esam_cfg = rep.get('esam', {}) or {}
        if esam_cfg.get('diagnostics_path'):
            diag_path = Path(esam_cfg['diagnostics_path'])
        else:
            diag_guess = diagnostics_root / f'seed{seed}__{rho_tag}.jsonl'
            if diag_guess.exists():
                diag_path = diag_guess

        sharp_mean = _mean_from_jsonl(diag_path, 'sharpness_proxy') if diag_path else None
        grad_mean = _mean_from_jsonl(diag_path, 'adapter_grad_norm') if diag_path else None

        frac = float(frac_tag.replace('p', '.'))
        run_dir = report_path.parent
        rows.append({
            'dataset': dataset,
            'train_fraction': frac,
            'seed': seed,
            'rho': rho,
            'start_epoch': start_epoch,
            'validation_metric': float(mval['score']),
            'test_metric': float(mtest['score']),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_val_gap': train_val_gap,
            'sharpness_proxy_mean': sharp_mean,
            'adapter_grad_norm_mean': grad_mean,
            'runtime': rep.get('time', ''),
            'runtime_seconds': _runtime_seconds(rep.get('time', '')),
            'run_id': str(rel.parent),
            'checkpoint_path': str(run_dir / 'checkpoint.pt'),
            'report_path': str(report_path),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--results-root', type=Path, required=True)
    ap.add_argument('--dataset', type=str, default='homesite-insurance')
    ap.add_argument('--warmup-epoch', type=int, default=5)
    args = ap.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)
    diag_root = args.results_root / 'diagnostics'

    df = collect_rows(args.run_root, diag_root)
    if df.empty:
        raise SystemExit('No runs found')
    df = df[df['dataset'] == args.dataset].copy()
    if df.empty:
        raise SystemExit(f'No runs found for dataset={args.dataset}')

    df = df.sort_values(['dataset', 'train_fraction', 'seed', 'start_epoch', 'rho']).reset_index(drop=True)
    df.to_csv(args.results_root / 'low_data_results.csv', index=False)

    # Refined rho sweep at start_epoch=0
    sweep = (
        df[df['start_epoch'] == 0]
        .groupby(['dataset', 'train_fraction', 'rho'])
        .agg(
            val_mean=('validation_metric', 'mean'),
            val_std=('validation_metric', 'std'),
            test_mean=('test_metric', 'mean'),
            test_std=('test_metric', 'std'),
            n=('seed', 'count'),
        )
        .reset_index()
        .sort_values(['train_fraction', 'rho'])
    )
    sweep.to_csv(args.results_root / 'esam_rho_refined_sweep.csv', index=False)

    # Paired deltas: baseline vs best ESAM by validation, per (fraction, seed), start_epoch=0
    base = df[(df['start_epoch'] == 0) & (df['rho'] == 0.0)].copy()
    nonzero = df[(df['start_epoch'] == 0) & (df['rho'] > 0.0)].copy()
    best_esam = (
        nonzero.sort_values(['dataset', 'train_fraction', 'seed', 'validation_metric'], ascending=[True, True, True, False])
        .groupby(['dataset', 'train_fraction', 'seed'], as_index=False)
        .head(1)
    )
    merged = base.merge(
        best_esam,
        on=['dataset', 'train_fraction', 'seed'],
        how='inner',
        suffixes=('_base', '_esam'),
    )
    deltas = pd.DataFrame({
        'dataset': merged['dataset'],
        'train_fraction': merged['train_fraction'],
        'seed': merged['seed'],
        'rho_selected': merged['rho_esam'],
        'val_base': merged['validation_metric_base'],
        'val_esam': merged['validation_metric_esam'],
        'val_delta_esam_minus_base': merged['validation_metric_esam'] - merged['validation_metric_base'],
        'test_base': merged['test_metric_base'],
        'test_esam': merged['test_metric_esam'],
        'test_delta_esam_minus_base': merged['test_metric_esam'] - merged['test_metric_base'],
    })
    deltas.to_csv(args.results_root / 'low_data_paired_deltas.csv', index=False)

    # Warmup ablation for nonzero rho: compare start0 vs warmup-epoch
    warm = df[(df['rho'] > 0.0) & (df['start_epoch'].isin([0, args.warmup_epoch]))].copy()
    warm_cmp = (
        warm.groupby(['dataset', 'train_fraction', 'rho', 'start_epoch'])
        .agg(
            val_mean=('validation_metric', 'mean'),
            test_mean=('test_metric', 'mean'),
            n=('seed', 'count'),
        )
        .reset_index()
        .sort_values(['train_fraction', 'rho', 'start_epoch'])
    )
    warm_cmp.to_csv(args.results_root / 'esam_warmup_ablation.csv', index=False)

    # Markdown report.
    best_global = (
        nonzero.groupby('rho')['validation_metric'].mean().sort_values(ascending=False)
    )
    best_rho = float(best_global.index[0]) if len(best_global) else None

    lines = []
    lines.append('# ESAM Low-Data Report (Homesite)')
    lines.append('')
    lines.append(f'- Total runs parsed: {len(df)}')
    lines.append(f'- Train fractions: {sorted(df.train_fraction.unique().tolist())}')
    lines.append(f'- Seeds: {sorted(df.seed.unique().tolist())}')
    lines.append(f'- Rhos observed: {sorted(df.rho.unique().tolist())}')
    lines.append(f'- Warmup epochs observed: {sorted(df.start_epoch.unique().tolist())}')
    lines.append('')
    lines.append('## Refined Rho Sweep (start_epoch=0)')
    lines.append(sweep.to_markdown(index=False))
    lines.append('')
    lines.append('## Paired Baseline vs Best-ESAM Deltas')
    lines.append(deltas.to_markdown(index=False) if not deltas.empty else 'No paired rows found.')
    lines.append('')
    lines.append('## Warmup Ablation')
    lines.append(warm_cmp.to_markdown(index=False) if not warm_cmp.empty else 'No warmup rows found.')
    lines.append('')
    lines.append('## Recommendation')
    if best_rho is None:
        lines.append('- No nonzero ESAM runs found.')
    else:
        lines.append(f'- Validation-only best global ESAM rho in this block: **{best_rho}**.')
        lines.append('- Use this rho for the next controlled comparison and keep baseline unchanged.')

    (args.results_root / 'ESAM_LOW_DATA_REPORT.md').write_text('\n'.join(lines))
    print(f'[done] {args.results_root / "low_data_results.csv"}')
    print(f'[done] {args.results_root / "ESAM_LOW_DATA_REPORT.md"}')


if __name__ == '__main__':
    main()
