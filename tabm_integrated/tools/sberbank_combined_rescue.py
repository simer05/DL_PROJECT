from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import tomli_w

TOOLS = Path(__file__).resolve().parent
PAPER = TOOLS.parent / 'paper'
EXP_ROOT = PAPER / 'exp' / 'integrated'
DATASET = 'sberbank-housing'
WAVE = 'rescue_sberbank_combined'
CONFIRM_WAVE = 'rescue_sberbank_combined_confirm'
MANIFEST = EXP_ROOT / 'manifest_sberbank_combined_rescue_seed0.txt'
CONFIRM_MANIFEST = EXP_ROOT / 'manifest_sberbank_combined_rescue_final.txt'
SELECTED_PATH = PAPER / 'exp' / 'sberbank_combined_rescue_selected.csv'
FINAL_SELECTED = PAPER / 'exp' / 'selected_integrated_configs.csv'

sys.path.insert(0, str(TOOLS))
import aggregate_integrated_results as agg  # noqa: E402
import generate_integrated_configs as gen  # noqa: E402


def ftag(x: float | int) -> str:
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return '0'
    return f'{x:g}'.replace('-', 'm').replace('.', 'p')


def write_config(wave: str, variant: str, cfg: dict[str, Any], seed: int) -> Path:
    cfg = json.loads(json.dumps(cfg))
    cfg['seed'] = int(seed)
    if cfg.get('model', {}).get('mfb'):
        cfg['model']['mfb']['mask_seed'] = int(seed)
    path = EXP_ROOT / wave / DATASET / f'{variant}-evaluation' / f'{seed}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(cfg))
    return path


def generate() -> None:
    paths: list[Path] = []
    for rank, noise, lam in itertools.product([1, 2, 4], [0.0, 1e-5, 1e-4, 1e-3], [0.001, 0.005, 0.01, 0.02]):
        cfg = gen.base_config(DATASET, 0)
        gen.apply_rla(cfg, rank=rank, noise=noise)
        gen.apply_cf_fisd(cfg, DATASET, lam=lam)
        variant = f'sb_comb_rla_cf_r{rank}_n{ftag(noise)}_l{ftag(lam)}'
        paths.append(write_config(WAVE, variant, cfg, 0))
    for rank, noise, rho in itertools.product([1, 2, 4], [0.0, 1e-5, 1e-4, 1e-3], [0.00025, 0.0005, 0.001, 0.0025]):
        cfg = gen.base_config(DATASET, 0)
        gen.apply_rla(cfg, rank=rank, noise=noise)
        gen.apply_esam(cfg, rho=rho)
        variant = f'sb_comb_rla_esam_r{rank}_n{ftag(noise)}_rho{ftag(rho)}'
        paths.append(write_config(WAVE, variant, cfg, 0))
    for rank, noise, keep, lam in itertools.product([1, 2, 4], [1e-5, 1e-4, 1e-3], [0.9, 0.95, 0.975], [0.001, 0.005, 0.01]):
        cfg = gen.base_config(DATASET, 0)
        gen.apply_rla(cfg, rank=rank, noise=noise)
        gen.apply_mfb(cfg, keep=keep)
        gen.apply_cf_fisd(cfg, DATASET, lam=lam)
        variant = f'sb_comb_rla_mfb_cf_r{rank}_n{ftag(noise)}_k{ftag(keep)}_l{ftag(lam)}'
        paths.append(write_config(WAVE, variant, cfg, 0))
    MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in paths) + '\n')
    print(f'{MANIFEST.relative_to(PAPER)} configs={len(paths)}')


def signed_delta(value: float, baseline: float, direction: str) -> tuple[float, float]:
    if direction == 'lower':
        d = baseline - value
        return d, 100.0 * d / baseline if baseline else float('nan')
    d = value - baseline
    return d, 100.0 * d / abs(baseline) if baseline else float('nan')


def collect_candidates(wave: str, seed: int = 0) -> list[dict[str, Any]]:
    rows = agg.collect_rows()
    baselines = {
        r['inference_mode']: r
        for r in rows
        if r['wave'] == 'final'
        and r['dataset'] == DATASET
        and r['variant'] == 'baseline_plr'
        and r['seed'] == seed
        and not r['failure']
    }
    missing = [x for x in ['mean', 'best-head', 'greedy-heads'] if x not in baselines]
    if missing:
        raise RuntimeError(f'missing matched baselines for {missing}')
    out: list[dict[str, Any]] = []
    for r in rows:
        if r['wave'] != wave or r['dataset'] != DATASET or r['seed'] != seed or r['failure']:
            continue
        b = baselines[r['inference_mode']]
        d, pct = signed_delta(r['validation_metric'], b['validation_metric'], r['direction'])
        e = dict(r)
        e['matched_validation_baseline'] = b['validation_metric']
        e['validation_delta'] = d
        e['validation_percent_delta'] = pct
        e['validation_status'] = 'validation_win' if d > 1e-12 else ('tie' if abs(d) <= 1e-12 else 'validation_loss')
        out.append(e)
    out.sort(key=lambda r: (r['validation_delta'], r['validation_score']), reverse=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def select_seed0() -> None:
    candidates = collect_candidates(WAVE, 0)
    top = candidates[:20]
    write_csv(SELECTED_PATH, top)
    winners = [r for r in candidates if r['validation_delta'] > 1e-12]
    confirm = winners[:2]
    paths: list[Path] = []
    for row in confirm:
        cfg = tomllib.loads((PAPER / row['config_path']).read_text())
        for seed in [0, 1, 2]:
            paths.append(write_config(CONFIRM_WAVE, row['variant'], cfg, seed))
    CONFIRM_MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in paths) + ('\n' if paths else ''))
    print(f'{SELECTED_PATH.relative_to(PAPER)} top_rows={len(top)} validation_winners={len(winners)}')
    print(f'{CONFIRM_MANIFEST.relative_to(PAPER)} configs={len(paths)}')
    for row in top[:5]:
        print(row['variant'], row['inference_mode'], row['validation_metric'], row['matched_validation_baseline'], row['validation_delta'], row['validation_status'])


def test_stats_for_variant(variant: str, inference: str) -> dict[str, Any] | None:
    rows = agg.collect_rows()
    baselines = [
        r for r in rows
        if r['wave'] == 'final'
        and r['dataset'] == DATASET
        and r['variant'] == 'baseline_plr'
        and r['inference_mode'] == inference
        and not r['failure']
    ]
    rs = [
        r for r in rows
        if r['wave'] == CONFIRM_WAVE
        and r['dataset'] == DATASET
        and r['variant'] == variant
        and r['inference_mode'] == inference
        and not r['failure']
    ]
    if len(rs) != 3 or len(baselines) != 3:
        return None
    import statistics
    rs = sorted(rs, key=lambda r: r['seed'])
    baselines = sorted(baselines, key=lambda r: r['seed'])
    mean = statistics.mean(r['test_metric'] for r in rs)
    std = statistics.stdev(r['test_metric'] for r in rs)
    base_mean = statistics.mean(r['test_metric'] for r in baselines)
    base_std = statistics.stdev(r['test_metric'] for r in baselines)
    delta, pct = signed_delta(mean, base_mean, rs[0]['direction'])
    status = agg.status_for(delta, base_std, 3, False)
    return {
        'variant': variant,
        'inference_mode': inference,
        'mean': mean,
        'std': std,
        'baseline_mean': base_mean,
        'baseline_std': base_std,
        'absolute_delta': delta,
        'percent_delta': pct,
        'status': status,
    }


def update_final_if_win() -> None:
    selected = []
    if SELECTED_PATH.exists() and SELECTED_PATH.read_text().strip():
        with SELECTED_PATH.open() as f:
            for row in csv.DictReader(f):
                if float(row['validation_delta']) > 1e-12:
                    selected.append(row)
    selected = selected[:2]
    results = []
    chosen = None
    for row in selected:
        stats = test_stats_for_variant(row['variant'], row['inference_mode'])
        if stats is None:
            continue
        result = dict(row) | stats
        results.append(result)
        if chosen is None and stats['status'] in {'clear_win', 'weak_win'}:
            chosen = result
    write_csv(PAPER / 'exp' / 'sberbank_combined_rescue_confirmed.csv', results)
    if chosen is None:
        print('no_confirmed_win')
        for r in results:
            print(r['variant'], r['inference_mode'], r['absolute_delta'], r['status'])
        return

    variant = chosen['variant']
    # Copy confirmation outputs into final best_combined slots.
    for seed in [0, 1, 2]:
        src_cfg = EXP_ROOT / CONFIRM_WAVE / DATASET / f'{variant}-evaluation' / f'{seed}.toml'
        dst_cfg = EXP_ROOT / 'final' / DATASET / 'best_combined-evaluation' / f'{seed}.toml'
        dst_cfg.write_text(src_cfg.read_text())
        for suffix in ['-evaluation', '-best-head-evaluation', '-greedy-heads-evaluation']:
            src = EXP_ROOT / CONFIRM_WAVE / DATASET / f'{variant}{suffix}' / str(seed)
            dst = EXP_ROOT / 'final' / DATASET / f'best_combined{suffix}' / str(seed)
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst)
    # Update validation-selected configs.
    rows = []
    with FINAL_SELECTED.open() as f:
        for row in csv.DictReader(f):
            if row['dataset'] == DATASET and row['final_variant'] == 'best_combined':
                row.update({
                    'source_variant': chosen['variant'],
                    'source_wave': WAVE,
                    'inference_mode': chosen['inference_mode'],
                    'validation_metric': chosen['validation_metric'],
                    'validation_score': chosen['validation_score'],
                    'matched_validation_baseline': str(chosen['matched_validation_baseline']),
                    'validation_delta': str(chosen['validation_delta']),
                    'validation_percent_delta': str(chosen['validation_percent_delta']),
                    'validation_status': chosen['validation_status'],
                    'source_config_path': chosen['config_path'],
                })
            rows.append(row)
    write_csv(FINAL_SELECTED, rows)
    print('updated_final_best_combined', chosen['variant'], chosen['inference_mode'], chosen['absolute_delta'], chosen['status'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['generate', 'select-seed0', 'update-final-if-win'])
    args = parser.parse_args()
    if args.stage == 'generate':
        generate()
    elif args.stage == 'select-seed0':
        select_seed0()
    elif args.stage == 'update-final-if-win':
        update_final_if_win()


if __name__ == '__main__':
    main()
