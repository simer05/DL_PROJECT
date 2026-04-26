"""Aggregate ecom-offers-only RLA sweep and select by validation AUROC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / 'paper' / 'exp' / 'rla' / 'ecom-offers'
OUT = REPO_ROOT / 'paper' / 'exp' / 'rla' / '_aggregated'
PREFIX = 'rla_ecom_sweep_'
MODES = ['mean', 'best-head', 'greedy-heads']
EPS = 1e-12


def eval_dir(variant: str, mode: str) -> Path:
    suffix = {
        'mean': '-evaluation',
        'best-head': '-best-head-evaluation',
        'greedy-heads': '-greedy-heads-evaluation',
    }[mode]
    return ROOT / f'{variant}{suffix}'


def baseline_for(variant: str) -> str:
    if '_k64_' in variant:
        return 'baseline_plr_k64'
    if '_fp32_' in variant:
        return 'baseline_plr_fp32'
    return 'baseline_plr'


def read_report(seed_dir: Path) -> dict | None:
    if not (seed_dir / 'DONE').exists() or not (seed_dir / 'report.json').exists():
        return None
    try:
        report = json.loads((seed_dir / 'report.json').read_text())
    except Exception:
        return None
    if report.get('failure'):
        return None
    if report.get('gpu_name') != 'NVIDIA GeForce RTX 4090':
        return None
    if report.get('amp_dtype') in (None, '', 'unknown'):
        return None
    return report


def metadata(report: dict) -> tuple:
    config = report.get('config') or {}
    model = config.get('model') or {}
    data = config.get('data') or {}
    return (
        report.get('gpu_name'),
        report.get('amp_dtype'),
        bool(config.get('amp', False)),
        model.get('k'),
        data.get('path'),
    )


def auc(report: dict, part: str) -> float:
    return float(((report.get('metrics') or {}).get(part) or {})['roc-auc'])


def score(report: dict, part: str) -> float:
    return float(((report.get('metrics') or {}).get(part) or {})['score'])


def comparison(variant: str, mode: str, seeds: tuple[int, ...]) -> tuple[dict, list[dict]] | None:
    baseline = baseline_for(variant)
    rows = []
    audit = []
    for seed in seeds:
        b_dir = eval_dir(baseline, mode) / str(seed)
        s_dir = eval_dir(variant, mode) / str(seed)
        b = read_report(b_dir)
        s = read_report(s_dir)
        if b is None or s is None or metadata(b) != metadata(s):
            return None
        rows.append((b, s))
        for role, var, seed_dir, report in [
            ('baseline', baseline, b_dir, b),
            ('selected', variant, s_dir, s),
        ]:
            audit.append(
                {
                    'dataset': 'ecom-offers',
                    'role': role,
                    'variant': var,
                    'inference_mode': mode,
                    'seed': seed,
                    'seed_path': str(seed_dir),
                    'DONE_present': (seed_dir / 'DONE').exists(),
                    'report_json_present': (seed_dir / 'report.json').exists(),
                    'gpu_name': report.get('gpu_name'),
                    'amp_dtype': report.get('amp_dtype'),
                    'val_metric': auc(report, 'val'),
                    'test_metric': auc(report, 'test'),
                    'failure_block_absent': not bool(report.get('failure')),
                }
            )
    val_base = mean(auc(b, 'val') for b, _ in rows)
    val_selected = mean(auc(s, 'val') for _, s in rows)
    test_base = mean(auc(b, 'test') for b, _ in rows)
    test_selected = mean(auc(s, 'test') for _, s in rows)
    val_score_delta = mean(score(s, 'val') - score(b, 'val') for b, s in rows)
    test_score_delta = mean(score(s, 'test') - score(b, 'test') for b, s in rows)
    delta = test_selected - test_base
    return (
        {
            'dataset': 'ecom-offers',
            'matched_baseline': f'{baseline}:{mode}',
            'selected_config': f'{variant}:{mode}',
            'validation_baseline': val_base,
            'validation_selected': val_selected,
            'test_baseline': test_base,
            'test_selected': test_selected,
            'delta': delta,
            'percent_delta': 100.0 * delta / abs(test_base),
            'n_seeds': len(seeds),
            'claim_status': (
                'win'
                if len(seeds) == 3 and val_score_delta > EPS and test_score_delta > EPS
                else ('tie' if abs(test_score_delta) <= EPS else 'loss')
            ),
            'metric': 'roc-auc',
            'seeds': ' '.join(str(x) for x in seeds),
            'validation_score_delta': val_score_delta,
            'test_score_delta': test_score_delta,
        },
        audit,
    )


def variants() -> list[str]:
    return sorted(
        p.name[: -len('-evaluation')]
        for p in ROOT.glob(f'{PREFIX}*-evaluation')
        if not p.name.endswith('-best-head-evaluation')
        and not p.name.endswith('-greedy-heads-evaluation')
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write('')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=2)
    args = parser.parse_args()

    smoke_rows = []
    for variant in variants():
        for mode in MODES:
            result = comparison(variant, mode, (0,))
            if result is not None:
                row, _ = result
                smoke_rows.append(row)
    smoke_rows.sort(key=lambda x: (x['validation_score_delta'], x['test_score_delta']), reverse=True)
    selected = smoke_rows[: args.top_k]
    write_csv(OUT / 'ecom_rla_sweep_smoke.csv', smoke_rows)
    write_csv(OUT / 'ecom_rla_sweep_selected.csv', selected)

    final_rows = []
    final_audit = []
    for row in selected:
        variant, mode = row['selected_config'].split(':', 1)
        result = comparison(variant, mode, (0, 1, 2))
        if result is not None:
            final, audit = result
            final_rows.append(final)
            final_audit.extend(audit)
    final_rows.sort(key=lambda x: (x['validation_score_delta'], x['test_score_delta']), reverse=True)
    write_csv(OUT / 'ecom_rla_sweep_final.csv', final_rows)
    write_csv(OUT / 'ecom_rla_sweep_audit_final.csv', final_audit)
    print('selected smoke:')
    for row in selected:
        print(row)
    print('final:')
    for row in final_rows:
        print(row)


if __name__ == '__main__':
    main()
