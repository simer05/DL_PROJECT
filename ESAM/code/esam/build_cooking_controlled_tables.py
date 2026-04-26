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


def collect_rows(run_root: Path, diagnostics_root: Path, dataset_filter: str) -> pd.DataFrame:
    rows = []
    for report_path in sorted(run_root.glob('**/report.json')):
        rel = report_path.relative_to(run_root)
        parts = rel.parts
        if len(parts) < 6:
            continue

        dataset = parts[0]
        if dataset != dataset_filter:
            continue
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

        esam_cfg = rep.get('esam', {}) or {}
        diag_path = None
        if esam_cfg.get('diagnostics_path'):
            diag_path = Path(esam_cfg['diagnostics_path'])
        if diag_path is None or not diag_path.exists():
            diag_guess = diagnostics_root / f'seed{seed}__{rho_tag}.jsonl'
            if diag_guess.exists():
                diag_path = diag_guess

        sharp_mean = _mean_from_jsonl(diag_path, 'sharpness_proxy') if diag_path else None
        grad_mean = _mean_from_jsonl(diag_path, 'adapter_grad_norm') if diag_path else None

        frac = float(frac_tag.replace('p', '.'))
        run_dir = report_path.parent
        method = 'baseline' if rho == 0.0 else 'esam'

        rows.append({
            'dataset': dataset,
            'train_fraction': frac,
            'seed': seed,
            'method': method,
            'rho': rho,
            'val_metric': float(mval['score']),
            'test_metric': float(mtest['score']),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_val_gap': train_val_gap,
            'sharpness_proxy_mean': sharp_mean,
            'adapter_grad_norm_mean': grad_mean,
            'runtime': rep.get('time', ''),
            'runtime_seconds': _runtime_seconds(rep.get('time', '')),
            'checkpoint_path': str(run_dir / 'checkpoint.pt'),
            'run_id': str(rel.parent),
            'start_epoch': start_epoch,
            'report_path': str(report_path),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['dataset', 'train_fraction', 'seed', 'rho']).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--results-root', type=Path, required=True)
    ap.add_argument('--dataset', type=str, default='cooking-time')
    args = ap.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)
    diag_root = args.results_root / 'diagnostics'

    df = collect_rows(args.run_root, diag_root, args.dataset)
    if df.empty:
        raise SystemExit(f'No runs found for dataset={args.dataset}')

    out_results = args.results_root / 'cooking_time_controlled_results.csv'
    df.to_csv(out_results, index=False)

    base = df[df['rho'] == 0.0].copy()
    esam = df[df['rho'] == 0.0025].copy()
    merged = base.merge(
        esam,
        on=['dataset', 'train_fraction', 'seed', 'start_epoch'],
        suffixes=('_base', '_esam'),
        how='inner',
    )
    paired = pd.DataFrame({
        'dataset': merged['dataset'],
        'train_fraction': merged['train_fraction'],
        'seed': merged['seed'],
        'rho_esam': merged['rho_esam'],
        'val_base': merged['val_metric_base'],
        'val_esam': merged['val_metric_esam'],
        'val_delta_esam_minus_base': merged['val_metric_esam'] - merged['val_metric_base'],
        'test_base': merged['test_metric_base'],
        'test_esam': merged['test_metric_esam'],
        'test_delta_esam_minus_base': merged['test_metric_esam'] - merged['test_metric_base'],
        'train_val_gap_base': merged['train_val_gap_base'],
        'train_val_gap_esam': merged['train_val_gap_esam'],
        'sharpness_proxy_mean_esam': merged['sharpness_proxy_mean_esam'],
        'adapter_grad_norm_mean_esam': merged['adapter_grad_norm_mean_esam'],
        'runtime_seconds_base': merged['runtime_seconds_base'],
        'runtime_seconds_esam': merged['runtime_seconds_esam'],
        'run_id_base': merged['run_id_base'],
        'run_id_esam': merged['run_id_esam'],
        'checkpoint_path_base': merged['checkpoint_path_base'],
        'checkpoint_path_esam': merged['checkpoint_path_esam'],
    }).sort_values(['train_fraction', 'seed'])

    out_paired = args.results_root / 'cooking_time_paired_deltas.csv'
    paired.to_csv(out_paired, index=False)

    lines = []
    lines.append('# Cooking-Time Controlled ESAM Report')
    lines.append('')
    lines.append(f'- Dataset: `{args.dataset}`')
    lines.append(f'- Runs parsed: {len(df)}')
    lines.append(f'- Fractions: {sorted(df.train_fraction.unique().tolist())}')
    lines.append(f'- Seeds: {sorted(df.seed.unique().tolist())}')
    lines.append('')

    summary = (
        df.groupby(['train_fraction', 'method'])
        .agg(
            val_mean=('val_metric', 'mean'),
            val_std=('val_metric', 'std'),
            test_mean=('test_metric', 'mean'),
            test_std=('test_metric', 'std'),
            train_loss_mean=('train_loss', 'mean'),
            val_loss_mean=('val_loss', 'mean'),
            train_val_gap_mean=('train_val_gap', 'mean'),
            runtime_s_mean=('runtime_seconds', 'mean'),
            n=('seed', 'count'),
        )
        .reset_index()
        .sort_values(['train_fraction', 'method'])
    )

    wins = (
        paired.groupby('train_fraction')['test_delta_esam_minus_base']
        .agg(
            mean_delta='mean',
            std_delta='std',
            wins=lambda s: int((s > 0).sum()),
            ties=lambda s: int((s == 0).sum()),
            losses=lambda s: int((s < 0).sum()),
            n='count',
        )
        .reset_index()
        .sort_values('train_fraction')
    )

    lines.append('## Mean ± Std by Fraction')
    lines.append(summary.to_markdown(index=False))
    lines.append('')
    lines.append('## Paired Test Deltas (ESAM - Baseline)')
    lines.append(paired[['train_fraction','seed','test_delta_esam_minus_base','val_delta_esam_minus_base']].to_markdown(index=False))
    lines.append('')
    lines.append('## Win Counts by Fraction')
    lines.append(wins.to_markdown(index=False))
    lines.append('')
    lines.append('## Interpretation')
    for _, r in wins.iterrows():
        frac = r['train_fraction']
        md = r['mean_delta']
        lines.append(f'- train_fraction={frac:.2f}: mean paired test delta={md:+.6f}, wins={int(r["wins"])} / {int(r["n"])}.')

    out_md = args.results_root / 'COOKING_TIME_ESAM_REPORT.md'
    out_md.write_text('\n'.join(lines))

    print(f'[done] {out_results}')
    print(f'[done] {out_paired}')
    print(f'[done] {out_md}')


if __name__ == '__main__':
    main()
