"""Build validation-only RLA claim table from strict matched runs."""

from __future__ import annotations

import csv
import json
import statistics
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'paper' / 'exp' / 'rla'
OUT = ROOT / '_aggregated'
DATASETS = ['sberbank-housing', 'ecom-offers', 'homesite-insurance', 'cooking-time', 'delivery-eta']
PRIMARY = {
    'homesite-insurance': 'roc-auc',
    'ecom-offers': 'roc-auc',
    'sberbank-housing': 'rmse',
    'cooking-time': 'rmse',
    'delivery-eta': 'rmse',
}
LOWER = {'rmse'}
EPS = 1e-12


def split_variant(name: str) -> tuple[str | None, str | None]:
    if name.endswith('-best-head-evaluation'):
        return name[: -len('-best-head-evaluation')], 'best-head'
    if name.endswith('-greedy-heads-evaluation'):
        return name[: -len('-greedy-heads-evaluation')], 'greedy-heads'
    if name.endswith('-evaluation'):
        return name[: -len('-evaluation')], 'mean'
    return None, None


def baseline_for(variant: str) -> str | None:
    if variant.startswith('baseline'):
        return None
    if 'mini_plr' in variant:
        return 'baseline_mini_plr'
    if '_fp32' in variant:
        return 'baseline_plr_fp32'
    if '_k64' in variant:
        return 'baseline_plr_k64'
    if 'plr' in variant:
        return 'baseline_plr'
    return 'baseline'


def metric(report: dict, dataset: str, part: str):
    return ((report.get('metrics') or {}).get(part) or {}).get(PRIMARY[dataset])


def score(report: dict, part: str):
    return ((report.get('metrics') or {}).get(part) or {}).get('score')


def mean(xs):
    values = [float(x) for x in xs if x is not None]
    return statistics.mean(values) if values else None


def compatible(a: dict, b: dict) -> bool:
    return (
        a['amp_dtype'] == b['amp_dtype']
        and a['amp'] == b['amp']
        and a['k'] == b['k']
        and a['data_path'] == b['data_path']
        and a['gpu_name'] == b['gpu_name']
    )


def metric_delta(dataset: str, selected, baseline):
    if selected is None or baseline is None:
        return None
    return baseline - selected if PRIMARY[dataset] in LOWER else selected - baseline


def percent_delta(delta, baseline):
    if delta is None or baseline in (None, 0):
        return None
    return 100.0 * delta / abs(baseline)


def load_rows() -> list[dict]:
    rows = []
    for ds_dir in ROOT.iterdir():
        if not ds_dir.is_dir() or ds_dir.name.startswith('_') or ds_dir.name not in DATASETS:
            continue
        dataset = ds_dir.name
        for eval_dir in ds_dir.iterdir():
            if not eval_dir.is_dir():
                continue
            variant, mode = split_variant(eval_dir.name)
            if variant is None:
                continue
            for seed_dir in eval_dir.iterdir():
                if not seed_dir.is_dir() or seed_dir.name not in {'0', '1', '2'}:
                    continue
                if not (seed_dir / 'DONE').exists() or not (seed_dir / 'report.json').exists():
                    continue
                try:
                    report = json.loads((seed_dir / 'report.json').read_text())
                except Exception:
                    continue
                if (
                    report.get('failure')
                    or report.get('gpu_name') != 'NVIDIA GeForce RTX 4090'
                    or report.get('amp_dtype') in (None, '', 'unknown')
                ):
                    continue
                config = report.get('config') or {}
                model = config.get('model') or {}
                data = config.get('data') or {}
                rows.append(
                    {
                        'dataset': dataset,
                        'variant': variant,
                        'mode': mode,
                        'seed': int(seed_dir.name),
                        'val': metric(report, dataset, 'val'),
                        'test': metric(report, dataset, 'test'),
                        'val_score': score(report, 'val'),
                        'test_score': score(report, 'test'),
                        'amp_dtype': report.get('amp_dtype'),
                        'amp': bool(config.get('amp', False)),
                        'k': model.get('k'),
                        'data_path': data.get('path'),
                        'gpu_name': report.get('gpu_name'),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant-prefix', default='')
    parser.add_argument('--out-name', default='rla_claim_table.csv')
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    by_key = {
        (r['dataset'], r['variant'], r['mode'], r['seed']): r
        for r in rows
    }
    selected = []
    for dataset in DATASETS:
        candidates = []
        variants = sorted(
            {
                r['variant']
                for r in rows
                if r['dataset'] == dataset and not r['variant'].startswith('baseline')
                and r['variant'].startswith(args.variant_prefix)
            }
        )
        for variant in variants:
            baseline = baseline_for(variant)
            if baseline is None:
                continue
            modes = sorted(
                {
                    r['mode']
                    for r in rows
                    if r['dataset'] == dataset and r['variant'] == variant
                }
            )
            for mode in modes:
                common_seeds = []
                for seed in (0, 1, 2):
                    base = by_key.get((dataset, baseline, mode, seed)) or by_key.get(
                        (dataset, baseline, 'mean', seed)
                    )
                    candidate = by_key.get((dataset, variant, mode, seed))
                    if base and candidate and compatible(base, candidate):
                        common_seeds.append(seed)
                if not common_seeds:
                    continue
                base_rows = [
                    by_key.get((dataset, baseline, mode, seed))
                    or by_key[(dataset, baseline, 'mean', seed)]
                    for seed in common_seeds
                ]
                candidate_rows = [
                    by_key[(dataset, variant, mode, seed)] for seed in common_seeds
                ]
                val_base = mean([r['val'] for r in base_rows])
                val_selected = mean([r['val'] for r in candidate_rows])
                test_base = mean([r['test'] for r in base_rows])
                test_selected = mean([r['test'] for r in candidate_rows])
                val_score_base = mean([r['val_score'] for r in base_rows])
                val_score_selected = mean([r['val_score'] for r in candidate_rows])
                test_score_base = mean([r['test_score'] for r in base_rows])
                test_score_selected = mean([r['test_score'] for r in candidate_rows])
                d = metric_delta(dataset, test_selected, test_base)
                candidates.append(
                    {
                        'dataset': dataset,
                        'matched_baseline': baseline,
                        'selected_config': f'{variant}:{mode}',
                        'validation_baseline': val_base,
                        'validation_selected': val_selected,
                        'test_baseline': test_base,
                        'test_selected': test_selected,
                        'delta': d,
                        'percent_delta': percent_delta(d, test_base),
                        'n_seeds': len(common_seeds),
                        'metric': PRIMARY[dataset],
                        'seeds': ' '.join(str(x) for x in common_seeds),
                        'validation_score_delta': (
                            None
                            if val_score_selected is None or val_score_base is None
                            else val_score_selected - val_score_base
                        ),
                        'test_score_delta': (
                            None
                            if test_score_selected is None or test_score_base is None
                            else test_score_selected - test_score_base
                        ),
                    }
                )
        if candidates:
            candidates.sort(
                key=lambda x: (
                    x['n_seeds'] == 3,
                    x['validation_score_delta']
                    if x['validation_score_delta'] is not None
                    else float('-inf'),
                ),
                reverse=True,
            )
            best = candidates[0]
            val_delta = best['validation_score_delta']
            test_delta = best['test_score_delta']
            if (
                best['n_seeds'] == 3
                and val_delta is not None
                and test_delta is not None
                and val_delta > EPS
                and test_delta > EPS
            ):
                best['claim_status'] = 'win'
            elif test_delta is not None and abs(test_delta) <= EPS:
                best['claim_status'] = 'tie'
            else:
                best['claim_status'] = 'loss'
            selected.append(best)

    fields = [
        'dataset',
        'matched_baseline',
        'selected_config',
        'validation_baseline',
        'validation_selected',
        'test_baseline',
        'test_selected',
        'delta',
        'percent_delta',
        'n_seeds',
        'claim_status',
        'metric',
        'seeds',
        'validation_score_delta',
        'test_score_delta',
    ]
    filenames = [args.out_name]
    if args.out_name == 'rla_claim_table.csv':
        filenames.insert(0, 'rla_validation_selected.csv')
    for filename in filenames:
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    print(f'wrote {OUT / args.out_name}')
    for row in selected:
        print(f"{row['dataset']}: {row['selected_config']} ({row['claim_status']})")


if __name__ == '__main__':
    main()
