
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import tomli_w

PAPER = Path(__file__).resolve().parents[1] / 'paper'
EXP_ROOT = PAPER / 'exp' / 'integrated'
SUMMARY_PATH = PAPER / 'exp' / 'final_integrated_summary.csv'
AUDIT_PATH = PAPER / 'exp' / 'final_integrated_audit.csv'
REPORT_PATH = PAPER.parent / 'FINAL_EXPERIMENT_REPORT.md'
SELECTED_PATH = PAPER / 'exp' / 'selected_integrated_configs.csv'
FINAL_MANIFEST = EXP_ROOT / 'manifest_final.txt'

DATASET_ORDER = ['sberbank-housing', 'ecom-offers', 'homesite-insurance', 'cooking-time', 'delivery-eta']
FINAL_DISPLAY = ['baseline_plr', 'best_rla_only', 'best_esam_only', 'best_mfb_only', 'best_cf_fisd_only', 'best_combined']


def infer_inference_mode(path: Path) -> str:
    parent = path.parent.parent.name
    if parent.endswith('-best-head-evaluation'):
        return 'best-head'
    if parent.endswith('-greedy-heads-evaluation'):
        return 'greedy-heads'
    return 'mean'


def mean_config_dir_for_result_dir(run_dir: Path) -> Path:
    eval_dir = run_dir.parent
    name = eval_dir.name
    if name.endswith('-best-head-evaluation'):
        mean_name = name.removesuffix('-best-head-evaluation') + '-evaluation'
    elif name.endswith('-greedy-heads-evaluation'):
        mean_name = name.removesuffix('-greedy-heads-evaluation') + '-evaluation'
    else:
        mean_name = name
    return eval_dir.with_name(mean_name) / run_dir.name


def variant_from_result(path: Path) -> str:
    name = path.parent.parent.name
    for suffix in ['-best-head-evaluation', '-greedy-heads-evaluation', '-evaluation']:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def dataset_from_result(path: Path) -> str:
    return path.parent.parent.parent.name


def wave_from_result(path: Path) -> str:
    return path.parent.parent.parent.parent.name


def score_key(report: dict[str, Any]) -> tuple[str, str, str]:
    val = report.get('metrics', {}).get('val', {})
    if 'rmse' in val:
        return 'RMSE', 'rmse', 'lower'
    if 'roc-auc' in val:
        return 'AUROC', 'roc-auc', 'higher'
    if 'accuracy' in val:
        return 'accuracy', 'accuracy', 'higher'
    raise KeyError(f'Unknown metrics keys: {sorted(val)}')


def signed_delta(test_mean: float, baseline_mean: float, direction: str) -> tuple[float, float]:
    if direction == 'lower':
        delta = baseline_mean - test_mean
        pct = 100.0 * delta / baseline_mean if baseline_mean else float('nan')
    else:
        delta = test_mean - baseline_mean
        pct = 100.0 * delta / abs(baseline_mean) if baseline_mean else float('nan')
    return delta, pct


def status_for(delta: float, baseline_std: float, n: int, invalid: bool) -> str:
    if invalid or n != 3:
        return 'invalid'
    if abs(delta) <= 1e-12:
        return 'tie'
    if delta < 0:
        return 'loss'
    if abs(delta) > baseline_std:
        return 'clear_win'
    return 'weak_win'


def iter_reports(root: Path = EXP_ROOT):
    for report_path in root.rglob('report.json'):
        if not report_path.parent.joinpath('DONE').exists():
            continue
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            continue
        mean_run_dir = mean_config_dir_for_result_dir(report_path.parent)
        cfg_path = mean_run_dir.with_suffix('.toml')
        yield report_path, cfg_path, report


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for report_path, cfg_path, report in iter_reports():
        dataset = dataset_from_result(report_path)
        variant = variant_from_result(report_path)
        wave = wave_from_result(report_path)
        inference = infer_inference_mode(report_path)
        metric, key, direction = score_key(report)
        rows.append({
            'wave': wave,
            'dataset': dataset,
            'variant': variant,
            'seed': int(report.get('config', {}).get('seed', report_path.parent.name)),
            'metric': metric,
            'metric_key': key,
            'direction': direction,
            'validation_metric': float(report['metrics']['val'][key]),
            'validation_score': float(report['metrics']['val']['score']),
            'test_metric': float(report['metrics']['test'][key]),
            'inference_mode': inference,
            'config_path': str(cfg_path.relative_to(PAPER)) if cfg_path.exists() else str(cfg_path),
            'result_path': str(report_path.parent.relative_to(PAPER)),
            'failure': bool(report.get('failure')),
            'best_step': report.get('best_step'),
        })
    return rows


def candidate_family(variant: str) -> str | None:
    if variant == 'baseline_plr':
        return 'baseline_plr'
    if variant == 'rla_only' or variant.startswith('rla_rank'):
        return 'best_rla_only'
    if variant == 'esam_only' or variant.startswith('esam_rho'):
        return 'best_esam_only'
    if variant == 'mfb_only' or variant.startswith('mfb_keep'):
        return 'best_mfb_only'
    if variant == 'cf_fisd_only' or variant.startswith('cf_fisd_lambda'):
        return 'best_cf_fisd_only'
    if variant in {'all_four_combined','rla_esam','rla_mfb','rla_cf_fisd','esam_mfb','esam_cf_fisd','mfb_cf_fisd','all_minus_rla','all_minus_esam','all_minus_mfb','all_minus_cf_fisd'}:
        return 'best_combined'
    return None


def select_final_configs() -> None:
    rows = [r for r in collect_rows() if r['wave'] in {'smoke', 'sweeps'} and r['seed'] == 0 and not r['failure']]
    by = defaultdict(list)
    for row in rows:
        fam = candidate_family(row['variant'])
        if fam and fam != 'baseline_plr':
            by[(row['dataset'], fam)].append(row)
    selected = []
    for dataset in DATASET_ORDER:
        baseline_cfg = EXP_ROOT / 'smoke' / dataset / 'baseline_plr-evaluation' / '0.toml'
        if not baseline_cfg.exists():
            baseline_cfg = EXP_ROOT / 'baseline_fidelity' / dataset / 'baseline_plr-evaluation' / '0.toml'
        selected.append({'dataset': dataset, 'final_variant': 'baseline_plr', 'source_variant': 'baseline_plr', 'source_wave': 'smoke', 'inference_mode': 'mean', 'validation_metric': '', 'validation_score': '', 'source_config_path': str(baseline_cfg.relative_to(PAPER))})
        for fam in FINAL_DISPLAY[1:]:
            candidates = by.get((dataset, fam), [])
            if not candidates:
                raise RuntimeError(f'No selection candidates for {dataset}/{fam}')
            best = max(candidates, key=lambda r: r['validation_score'])
            selected.append({'dataset': dataset, 'final_variant': fam, 'source_variant': best['variant'], 'source_wave': best['wave'], 'inference_mode': best['inference_mode'], 'validation_metric': best['validation_metric'], 'validation_score': best['validation_score'], 'source_config_path': best['config_path']})

    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SELECTED_PATH.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    final_paths: list[Path] = []
    for row in selected:
        source_cfg = PAPER / row['source_config_path']
        cfg = tomllib.loads(source_cfg.read_text())
        for seed in [0, 1, 2]:
            cfg_seed = json.loads(json.dumps(cfg))
            cfg_seed['seed'] = seed
            if cfg_seed.get('model', {}).get('mfb'):
                cfg_seed['model']['mfb']['mask_seed'] = seed
            out = EXP_ROOT / 'final' / row['dataset'] / f"{row['final_variant']}-evaluation" / f'{seed}.toml'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tomli_w.dumps(cfg_seed))
            final_paths.append(out)
    FINAL_MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in final_paths) + '\n')
    print(f'selected configs: {SELECTED_PATH.relative_to(PAPER)}')
    print(f'final manifest: {FINAL_MANIFEST.relative_to(PAPER)} ({len(final_paths)} jobs)')


def summarize_final() -> list[dict[str, Any]]:
    selection = {}
    if SELECTED_PATH.exists():
        with SELECTED_PATH.open() as f:
            for row in csv.DictReader(f):
                selection[(row['dataset'], row['final_variant'])] = row
    rows = [r for r in collect_rows() if r['wave'] == 'final']
    wanted = set((d, v) for d in DATASET_ORDER for v in FINAL_DISPLAY)
    grouped = defaultdict(list)
    for row in rows:
        if (row['dataset'], row['variant']) not in wanted:
            continue
        sel = selection.get((row['dataset'], row['variant']))
        if sel and row['inference_mode'] != sel['inference_mode']:
            continue
        if row['variant'] == 'baseline_plr' and row['inference_mode'] != 'mean':
            continue
        grouped[(row['dataset'], row['variant'])].append(row)

    out = []
    baseline_stats = {}
    for dataset in DATASET_ORDER:
        vals = [r['test_metric'] for r in grouped.get((dataset, 'baseline_plr'), [])]
        baseline_stats[dataset] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0) if vals else (float('nan'), float('nan'))

    for dataset in DATASET_ORDER:
        for variant in FINAL_DISPLAY:
            rs = sorted(grouped.get((dataset, variant), []), key=lambda r: r['seed'])
            n = len(rs)
            invalid = n != 3 or any(r['failure'] for r in rs)
            metric = rs[0]['metric'] if rs else ''
            direction = rs[0]['direction'] if rs else ''
            val_mean = statistics.mean([r['validation_metric'] for r in rs]) if rs else float('nan')
            test_values = [r['test_metric'] for r in rs]
            test_mean = statistics.mean(test_values) if test_values else float('nan')
            test_std = statistics.stdev(test_values) if len(test_values) > 1 else 0.0
            base_mean, base_std = baseline_stats[dataset]
            delta, pct = signed_delta(test_mean, base_mean, direction) if rs else (float('nan'), float('nan'))
            status = 'baseline' if variant == 'baseline_plr' else status_for(delta, base_std, n, invalid)
            sel = selection.get((dataset, variant), {})
            out.append({
                'dataset': dataset,
                'variant': variant,
                'source_variant': sel.get('source_variant', variant),
                'metric': metric,
                'direction': direction,
                'validation_metric': val_mean,
                'test_metric': test_mean,
                'mean': test_mean,
                'std': test_std,
                'n_seeds': n,
                'baseline_mean': base_mean,
                'absolute_delta': delta,
                'percent_delta': pct,
                'precision': '3 seeds',
                'inference_mode': sel.get('inference_mode', 'mean' if variant == 'baseline_plr' else ''),
                'config_path': ';'.join(r['config_path'] for r in rs),
                'result_path': ';'.join(r['result_path'] for r in rs),
                'status': status,
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return 'nan'
        return f'{x:.6g}'
    return str(x)


def write_report(summary_rows: list[dict[str, Any]]) -> None:
    lines = ['# Integrated TabM final experiment report', '']
    lines += ['Official TabM PLR/PiecewiseLinearEmbeddings baseline configs are preserved exactly; variants add only module flags for RLA, ESAM, MFB, and CF-FISD.', '']
    lines += ['## Final 3-seed results', '']
    lines += ['| dataset | variant | source config | metric | inference | mean ± std | delta | status |', '|---|---|---|---|---|---:|---:|---|']
    for r in summary_rows:
        mean_std = f"{fmt(r['mean'])} ± {fmt(r['std'])}"
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['source_variant']} | {r['metric']} | {r['inference_mode']} | {mean_std} | {fmt(r['absolute_delta'])} | {r['status']} |")
    lines += ['', '## Validation-selected configs', '']
    if SELECTED_PATH.exists():
        lines += ['| dataset | final variant | selected sweep variant | inference | validation metric |', '|---|---|---|---|---:|']
        with SELECTED_PATH.open() as f:
            for row in csv.DictReader(f):
                lines.append(f"| {row['dataset']} | {row['final_variant']} | {row['source_variant']} | {row['inference_mode']} | {row['validation_metric']} |")
    lines += ['', '## Module wins vs baseline', '']
    for variant in FINAL_DISPLAY[1:]:
        wins = [r['dataset'] for r in summary_rows if r['variant'] == variant and r['status'] in {'clear_win','weak_win'}]
        lines.append(f"- `{variant}`: {', '.join(wins) if wins else 'none'}")
    REPORT_PATH.write_text('\n'.join(lines) + '\n')


def stage_wave(manifest: str) -> None:
    rows = collect_rows()
    print(f'aggregated complete reports: {len(rows)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['wave', 'select-final', 'final'], default='final')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    if args.stage == 'wave':
        stage_wave(args.manifest or '')
        return
    if args.stage == 'select-final':
        select_final_configs()
        return
    summary = summarize_final()
    write_csv(SUMMARY_PATH, summary)
    audit_rows = collect_rows()
    if audit_rows:
        write_csv(AUDIT_PATH, audit_rows)
    write_report(summary)
    print(f'summary: {SUMMARY_PATH.relative_to(PAPER)}')
    print(f'audit: {AUDIT_PATH.relative_to(PAPER)}')
    print(f'report: {REPORT_PATH.relative_to(PAPER.parent)}')


if __name__ == '__main__':
    main()
