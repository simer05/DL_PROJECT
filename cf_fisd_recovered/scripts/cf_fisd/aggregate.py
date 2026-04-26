from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import paramiko

GW_HOST, GW_USER = '172.21.26.100', 'simerjit001'
TGT_HOST, TGT_USER = 'aspire2antu.nscc.sg', 'simerjit'
PAPER_DIR_REMOTE = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'

DATASETS = (
    'sberbank-housing', 'ecom-offers', 'homesite-insurance',
    'cooking-time', 'delivery-eta',
)
INFERENCE_MODES = ('default', 'best_head', 'greedy_heads')


def connect(password: str):
    gw = paramiko.SSHClient()
    gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=password, timeout=30,
               allow_agent=False, look_for_keys=False)
    chan = gw.get_transport().open_channel('direct-tcpip', (TGT_HOST, 22), ('127.0.0.1', 0))
    tgt = paramiko.SSHClient()
    tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=password, sock=chan, timeout=30,
                allow_agent=False, look_for_keys=False)
    return gw, tgt


def variant_dir_for(mode: str, base: str) -> str:
    if mode == 'default':
        return base
    if mode == 'best_head':
        return base.replace('-evaluation', '-best-head-evaluation')
    if mode == 'greedy_heads':
        return base.replace('-evaluation', '-greedy-heads-evaluation')
    raise ValueError(mode)


def _read_toml_amp(sftp, toml_path: str) -> bool | None:
    try:
        with sftp.open(toml_path) as f:
            for raw in f.read().decode().splitlines():
                line = raw.split('#', 1)[0].strip()
                if line.startswith('amp') and '=' in line:
                    val = line.split('=', 1)[1].strip().lower()
                    if val in ('true', '1'):
                        return True
                    if val in ('false', '0'):
                        return False
    except (FileNotFoundError, IOError):
        return None
    return None


def fetch_reports(holder, ds: str, variant: str, n_seeds: int):
    sftp = holder['tgt'].open_sftp()
    rows = []
    base_remote = f'{PAPER_DIR_REMOTE}/exp/cf_fisd/tabred/{ds}/{variant}-evaluation'
    for mode in INFERENCE_MODES:
        eval_dir = variant_dir_for(mode, base_remote)
        for s in range(n_seeds):
            seed_dir = f'{eval_dir}/{s}'
            try:
                with sftp.open(f'{seed_dir}/report.json') as f:
                    rep = json.load(f)
                with sftp.open(f'{seed_dir}/DONE') as f:
                    pass
            except (FileNotFoundError, IOError):
                continue
            try:
                cfg_amp = bool(rep.get('config', {}).get('amp', False))
                rep_seed = int(rep.get('config', {}).get('seed', s))
            except Exception:
                cfg_amp = False
                rep_seed = s
            if rep_seed != s:
                print(f'  parity skip seed: {seed_dir}: report.seed={rep_seed} != dir={s}',
                      file=sys.stderr)
                continue
            toml_path = f'{base_remote}/{s}.toml' if mode == 'default' else None
            if toml_path is not None:
                toml_amp = _read_toml_amp(sftp, toml_path)
                if toml_amp is not None and toml_amp != cfg_amp:
                    print(f'  parity skip amp: {seed_dir}: report.amp={cfg_amp} != toml.amp={toml_amp}',
                          file=sys.stderr)
                    continue
            score = rep.get('metrics', {}).get('test', {}).get('score')
            if score is None:
                continue
            row = {
                'dataset': ds,
                'variant': variant,
                'inference_mode': mode,
                'seed': s,
                'test_score': float(score),
                'val_score': float(rep['metrics']['val']['score']),
                'amp': cfg_amp,
                'amp_dtype': rep.get('amp_dtype'),
                'best_step': rep.get('best_step'),
                'time': rep.get('time'),
                'n_parameters': rep.get('n_parameters'),
                'task_metric_value': _raw_metric_value(rep),
                'cf_fisd_lambda': rep.get('cf_fisd', {}).get('lambda') if isinstance(rep.get('cf_fisd'), dict) else None,
                'cf_fisd_variant': rep.get('cf_fisd', {}).get('variant') if isinstance(rep.get('cf_fisd'), dict) else None,
            }
            rows.append(row)
    sftp.close()
    return rows


def _raw_metric_value(report: dict):
    metrics = report.get('metrics', {}).get('test', {})
    for key in ('rmse', 'roc-auc', 'roc_auc', 'accuracy', 'auc'):
        if key in metrics:
            return float(metrics[key])
    return None


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        key = (r['dataset'], r['variant'], r['inference_mode'])
        grouped[key].append(r)
    out = []
    for (ds, var, mode), records in sorted(grouped.items()):
        scores = [r['test_score'] for r in records]
        raws = [r['task_metric_value'] for r in records if r['task_metric_value'] is not None]
        n = len(scores)
        mean = statistics.mean(scores) if scores else float('nan')
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        out.append({
            'dataset': ds,
            'variant': var,
            'inference_mode': mode,
            'n_seeds': n,
            'mean_score': mean,
            'std_score': std,
            'mean_raw_metric': statistics.mean(raws) if raws else float('nan'),
            'std_raw_metric': statistics.stdev(raws) if len(raws) > 1 else 0.0,
            'val_best_seed': max(records, key=lambda r: r['val_score'])['seed'] if records else None,
        })
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, summary: list[dict]):
    by_mode = defaultdict(list)
    for r in summary:
        by_mode[r['inference_mode']].append(r)
    parts = ['# CF-FISD aggregated results\n']
    for mode in INFERENCE_MODES:
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        parts.append(f'\n## Inference mode: `{mode}`\n\n')
        parts.append('| dataset | variant | n | mean score | std | raw mean | raw std |\n')
        parts.append('|---|---|---|---|---|---|---|\n')
        for r in sorted(rows, key=lambda r: (r['dataset'], r['variant'])):
            parts.append(
                f"| {r['dataset']} | {r['variant']} | {r['n_seeds']} "
                f"| {r['mean_score']:+.6f} | {r['std_score']:.6f} "
                f"| {r['mean_raw_metric']:+.6f} | {r['std_raw_metric']:.6f} |\n"
            )
    path.write_text(''.join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument(
        '--variants',
        nargs='+',
        default=['baseline_plr', 'hetero_raw_lam0.05', 'hetero_raw_lam0.1', 'hetero_raw_lam0.2'],
    )
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--out-dir', type=Path, default=Path('D:/TabM_PROJ/tabm_fork/paper/exp/cf_fisd/_aggregated'))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    password = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not password:
        print('ERROR: set NSCC_PW env var', file=sys.stderr); sys.exit(2)
    print('connecting...', flush=True)
    gw, tgt = connect(password)
    holder = {'gw': gw, 'tgt': tgt}

    all_rows = []
    for ds in args.datasets:
        for v in args.variants:
            rows = fetch_reports(holder, ds, v, args.n_seeds)
            print(f'{ds:<22} {v:<28} -> {len(rows)} rows')
            all_rows.extend(rows)

    write_csv(args.out_dir / 'long.csv', all_rows)
    summary = aggregate(all_rows)
    write_csv(args.out_dir / 'wide.csv', summary)
    write_markdown(args.out_dir / 'report.md', summary)
    print(f'wrote: {args.out_dir / "long.csv"}, wide.csv, report.md')

    try:
        holder['tgt'].close(); holder['gw'].close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
