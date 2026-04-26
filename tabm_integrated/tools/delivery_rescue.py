from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
import tomllib
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import tomli_w

TOOLS = Path(__file__).resolve().parent
PAPER = TOOLS.parent / 'paper'
EXP_ROOT = PAPER / 'exp' / 'integrated'
TARGETS_PATH = PAPER / 'exp' / 'rescue_targets.csv'
RESCUE_SELECTED_PATH = PAPER / 'exp' / 'rescue_selected_configs.csv'
SELECTED_PATH = PAPER / 'exp' / 'selected_integrated_configs.csv'
IND_MANIFEST = EXP_ROOT / 'manifest_delivery_rescue_individual.txt'
COMB_MANIFEST = EXP_ROOT / 'manifest_delivery_rescue_combined.txt'
FINAL_MANIFEST = EXP_ROOT / 'manifest_delivery_rescue_final.txt'
DATASET = 'delivery-eta'

sys.path.insert(0, str(TOOLS))
import generate_integrated_configs as gen  # noqa: E402
import aggregate_integrated_results as agg  # noqa: E402

MODULE_VARIANTS = {
    'RLA': 'best_rla_only',
    'ESAM': 'best_esam_only',
    'MFB': 'best_mfb_only',
    'CF-FISD': 'best_cf_fisd_only',
    'combined': 'best_combined',
}


def ftag(x: float | int | bool) -> str:
    if isinstance(x, bool):
        return 't' if x else 'f'
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return '0'
    return f'{x:g}'.replace('-', 'm').replace('.', 'p')


def start_epoch(cfg: dict[str, Any], frac: float) -> int:
    budget = int(cfg['n_epochs']) if int(cfg['n_epochs']) > 0 else 2 * int(cfg['patience'])
    return int(math.ceil(budget * frac))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_config(wave: str, variant: str, cfg: dict[str, Any], seed: int = 0) -> Path:
    cfg = deepcopy(cfg)
    cfg['seed'] = seed
    if cfg.get('model', {}).get('mfb'):
        cfg['model']['mfb']['mask_seed'] = seed
    path = EXP_ROOT / wave / DATASET / f'{variant}-evaluation' / f'{seed}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(cfg))
    return path


def apply_rla(cfg: dict[str, Any], rank: int, noise: float, lr: float, freeze: float) -> None:
    gen.apply_rla(cfg, rank=rank, noise=noise)
    cfg['model']['rla_first_only'] = False
    cfg['rla_adapter_lr_multiplier'] = lr
    cfg['rla_extra_paths_freeze_fraction'] = freeze


def apply_esam(cfg: dict[str, Any], rho: float, frac: float, adapter_only: bool) -> None:
    gen.apply_esam(cfg, rho=rho)
    cfg['esam_adapter_only'] = adapter_only
    cfg['esam_start_epoch'] = start_epoch(cfg, frac)


def apply_mfb(cfg: dict[str, Any], keep: float, frac: float, group_mode: str) -> None:
    gen.apply_mfb(cfg, keep=keep)
    cfg['model']['mfb']['start_epoch'] = start_epoch(cfg, frac)
    cfg['model']['mfb']['start_fraction'] = frac
    cfg['model']['mfb']['group_mode'] = group_mode
    cfg['model']['mfb']['categorical_handling'] = 'no_cat_drop'


def apply_cf(cfg: dict[str, Any], lam: float, mode: str) -> None:
    gen.apply_cf_fisd(cfg, DATASET, lam=lam)
    cf = cfg['cf_fisd']
    cf['start_epoch'] = start_epoch(cfg, 0.5)
    cf['start_fraction'] = 0.5
    cf['mode'] = mode
    if mode == 'softmax':
        cf['variant'] = 'softmax'
    elif mode == 'consensus_raw':
        cf['variant'] = 'raw'
        k = int(cfg['model']['k'])
        cf['member_groups'] = {name: list(range(k)) for name in cf['teacher_names']}
    else:
        raise ValueError(mode)


def generate_targets() -> None:
    rows = read_csv(PAPER / 'exp' / 'final_integrated_summary.csv')
    out = []
    for row in rows:
        if row['dataset'] != DATASET or row['variant'] == 'baseline_plr':
            continue
        module = {v: k for k, v in MODULE_VARIANTS.items()}[row['variant']]
        out.append({
            'priority': 0,
            'dataset': DATASET,
            'module': module,
            'final_variant': row['variant'],
            'current_source_variant': row['source_variant'],
            'current_inference_mode': row['inference_mode'],
            'metric': row['metric'],
            'direction': row['direction'],
            'matched_baseline_test_mean': row['matched_baseline_mean'],
            'current_test_mean': row['mean'],
            'current_absolute_delta': row['absolute_delta'],
            'current_percent_delta': row['percent_delta'],
            'current_status': row['status'],
            'rescue_reason': 'delivery_eta_priority' if row['status'] in {'clear_win', 'weak_win'} else 'matched_baseline_loss',
        })
    order = {'RLA': 0, 'ESAM': 1, 'MFB': 2, 'CF-FISD': 3, 'combined': 4}
    out.sort(key=lambda r: order[r['module']])
    write_csv(TARGETS_PATH, out)
    print(f'{TARGETS_PATH.relative_to(PAPER)} rows={len(out)} counts={dict(Counter(r["module"] for r in out))}')


def generate_individual() -> None:
    paths: list[Path] = []
    for rank, noise, lr, freeze in itertools.product([1, 2, 4], [0.0, 1e-5, 1e-4], [0.25, 0.5], [0.0, 0.5]):
        cfg = gen.base_config(DATASET, 0)
        apply_rla(cfg, rank, noise, lr, freeze)
        paths.append(write_config('rescue_delivery_individual', f'deliv_rla_r{rank}_n{ftag(noise)}_lr{ftag(lr)}_fr{ftag(freeze)}', cfg))
    for rho, frac, adapter_only in itertools.product([0.00025, 0.0005, 0.001], [0.25, 0.5], [True, False]):
        cfg = gen.base_config(DATASET, 0)
        apply_esam(cfg, rho, frac, adapter_only)
        paths.append(write_config('rescue_delivery_individual', f'deliv_esam_rho{ftag(rho)}_sf{ftag(frac)}_adapter{ftag(adapter_only)}', cfg))
    for keep, frac, group_mode in itertools.product([0.975, 0.99], [0.5, 0.75], ['numerical_only', 'per_member']):
        cfg = gen.base_config(DATASET, 0)
        apply_mfb(cfg, keep, frac, group_mode)
        paths.append(write_config('rescue_delivery_individual', f'deliv_mfb_k{ftag(keep)}_sf{ftag(frac)}_{group_mode}', cfg))
    for lam, mode in itertools.product([0.005, 0.01, 0.02], ['consensus_raw', 'softmax']):
        cfg = gen.base_config(DATASET, 0)
        apply_cf(cfg, lam, mode)
        paths.append(write_config('rescue_delivery_individual', f'deliv_cf_l{ftag(lam)}_{mode}', cfg))
    IND_MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in paths) + '\n')
    print(f'{IND_MANIFEST.relative_to(PAPER)} configs={len(paths)}')


def family(variant: str) -> str | None:
    if variant.startswith('deliv_rla_'):
        return 'RLA'
    if variant.startswith('deliv_esam_'):
        return 'ESAM'
    if variant.startswith('deliv_mfb_'):
        return 'MFB'
    if variant.startswith('deliv_cf_'):
        return 'CF-FISD'
    if variant.startswith('deliv_comb_'):
        return 'combined'
    return None


def delta(value: float, baseline: float, direction: str) -> tuple[float, float]:
    if direction == 'lower':
        d = baseline - value
        return d, 100.0 * d / baseline if baseline else float('nan')
    d = value - baseline
    return d, 100.0 * d / abs(baseline) if baseline else float('nan')


def seed0_baselines(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if row['wave'] == 'final' and row['dataset'] == DATASET and row['variant'] == 'baseline_plr' and row['seed'] == 0 and not row['failure']:
            out[row['inference_mode']] = row
    missing = [x for x in ['mean', 'best-head', 'greedy-heads'] if x not in out]
    if missing:
        raise RuntimeError(f'missing delivery seed-0 baselines: {missing}')
    return out


def candidate_rows(wave: str) -> list[dict[str, Any]]:
    rows = agg.collect_rows()
    bases = seed0_baselines(rows)
    out = []
    for row in rows:
        if row['wave'] != wave or row['dataset'] != DATASET or row['seed'] != 0 or row['failure']:
            continue
        mod = family(row['variant'])
        if mod is None:
            continue
        base = bases[row['inference_mode']]
        d, pct = delta(row['validation_metric'], base['validation_metric'], row['direction'])
        enriched = dict(row)
        enriched['module'] = mod
        enriched['final_variant'] = MODULE_VARIANTS[mod]
        enriched['matched_validation_baseline'] = base['validation_metric']
        enriched['validation_delta'] = d
        enriched['validation_percent_delta'] = pct
        enriched['validation_status'] = 'validation_win' if d > 1e-12 else ('close_no_validation_win' if pct >= -0.1 else 'no_validation_win')
        out.append(enriched)
    return out


def best_by_module(wave: str, modules: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows(wave):
        grouped[row['module']].append(row)
    best = {}
    for mod in modules:
        if not grouped.get(mod):
            raise RuntimeError(f'no completed candidates for {mod} in {wave}')
        best[mod] = max(grouped[mod], key=lambda r: (r['validation_delta'], r['validation_score']))
    return best


def copy_rla(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ['rla_rank', 'rla_first_only', 'rla_additive', 'rla_init', 'rla_base_preserve_noise']:
        if key in src['model']:
            dst['model'][key] = src['model'][key]
    for key in ['rla_adapter_lr_multiplier', 'rla_extra_paths_freeze_fraction']:
        if key in src:
            dst[key] = src[key]


def copy_esam(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ['use_esam', 'esam_rho', 'esam_eps', 'esam_adapter_only', 'esam_memberwise', 'esam_warmup_epochs', 'esam_start_epoch', 'esam_end_epoch', 'esam_log_diagnostics', 'esam_diagnostics_every']:
        if key in src:
            dst[key] = src[key]


def copy_cf(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst['cf_fisd'] = deepcopy(src['cf_fisd'])


def generate_combined() -> None:
    best = best_by_module('rescue_delivery_individual', ['RLA', 'ESAM', 'MFB', 'CF-FISD'])
    cfgs = {mod: tomllib.loads((PAPER / row['config_path']).read_text()) for mod, row in best.items()}
    combos = {
        'deliv_comb_rla_esam': ('RLA', 'ESAM'),
        'deliv_comb_rla_cf': ('RLA', 'CF-FISD'),
        'deliv_comb_esam_cf': ('ESAM', 'CF-FISD'),
        'deliv_comb_rla_esam_cf': ('RLA', 'ESAM', 'CF-FISD'),
    }
    paths = []
    for variant, mods in combos.items():
        cfg = gen.base_config(DATASET, 0)
        if 'RLA' in mods:
            copy_rla(cfg, cfgs['RLA'])
        if 'ESAM' in mods:
            copy_esam(cfg, cfgs['ESAM'])
        if 'CF-FISD' in mods:
            copy_cf(cfg, cfgs['CF-FISD'])
        paths.append(write_config('rescue_delivery_combined', variant, cfg))
    COMB_MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in paths) + '\n')
    print(f'{COMB_MANIFEST.relative_to(PAPER)} configs={len(paths)}')


def select_and_make_final() -> None:
    selected = best_by_module('rescue_delivery_individual', ['RLA', 'ESAM', 'MFB', 'CF-FISD'])
    selected.update(best_by_module('rescue_delivery_combined', ['combined']))
    rows = []
    for mod in ['RLA', 'ESAM', 'MFB', 'CF-FISD', 'combined']:
        row = selected[mod]
        rows.append({
            'dataset': DATASET,
            'module': mod,
            'final_variant': MODULE_VARIANTS[mod],
            'source_variant': row['variant'],
            'source_wave': row['wave'],
            'inference_mode': row['inference_mode'],
            'validation_metric': row['validation_metric'],
            'matched_validation_baseline': row['matched_validation_baseline'],
            'validation_delta': row['validation_delta'],
            'validation_percent_delta': row['validation_percent_delta'],
            'validation_status': row['validation_status'],
            'validation_score': row['validation_score'],
            'source_config_path': row['config_path'],
            'seed0_result_path': row['result_path'],
            'confirm_3seed': True,
        })
    write_csv(RESCUE_SELECTED_PATH, rows)

    selection = read_csv(SELECTED_PATH)
    repl = {(r['dataset'], r['final_variant']): r for r in rows}
    updated = []
    for row in selection:
        key = (row['dataset'], row['final_variant'])
        if key in repl:
            r = repl[key]
            new = dict(row)
            new.update({
                'source_variant': r['source_variant'],
                'source_wave': r['source_wave'],
                'inference_mode': r['inference_mode'],
                'validation_metric': str(r['validation_metric']),
                'validation_score': str(r['validation_score']),
                'matched_validation_baseline': str(r['matched_validation_baseline']),
                'validation_delta': str(r['validation_delta']),
                'validation_percent_delta': str(r['validation_percent_delta']),
                'validation_status': r['validation_status'],
                'source_config_path': r['source_config_path'],
            })
            updated.append(new)
        else:
            updated.append(row)
    write_csv(SELECTED_PATH, updated)

    final_paths = []
    for r in rows:
        cfg = tomllib.loads((PAPER / r['source_config_path']).read_text())
        for seed in [0, 1, 2]:
            final_paths.append(write_config('final', r['final_variant'], cfg, seed))
    FINAL_MANIFEST.write_text('\n'.join(str(p.relative_to(PAPER)) for p in final_paths) + '\n')
    print(f'{RESCUE_SELECTED_PATH.relative_to(PAPER)} rows={len(rows)}')
    print(f'{FINAL_MANIFEST.relative_to(PAPER)} configs={len(final_paths)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['targets', 'generate-individual', 'generate-combined', 'select-final'])
    args = parser.parse_args()
    if args.stage == 'targets':
        generate_targets()
    elif args.stage == 'generate-individual':
        generate_individual()
    elif args.stage == 'generate-combined':
        generate_combined()
    elif args.stage == 'select-final':
        select_and_make_final()


if __name__ == '__main__':
    main()
