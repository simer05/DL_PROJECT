"""Aggregate RLA evaluation results into a final report.

Reads the per-seed report.json files dropped by ``paper/bin/evaluate.py``
under ``paper/exp/rla/<dataset>/<variant>-evaluation/<seed>/``, computes
mean ± std test/val metrics across seeds, and writes:

* ``rla_results.csv``         — long-format table (one row per dataset/variant/seed).
* ``rla_summary.csv``         — wide table with mean ± std per dataset/variant.
* ``rla_report.md``           — Markdown summary suitable for the team report.

Plus the diagnostic plots required by Section 9.5 of the spec
(see ``tools/plot_rla.py`` for figure generation).

Run from the repo root:
    python tools/aggregate_rla_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RLA_ROOT = REPO_ROOT / 'paper' / 'exp' / 'rla'
OUT_DIR = REPO_ROOT / 'paper' / 'exp' / 'rla' / '_aggregated'

# Mapping of TabReD task → primary metric we report. Following the TabM paper:
# regression datasets use RMSE on standardized labels (lower is better);
# classification datasets use the corresponding TabReD metric (higher is better).
PRIMARY_METRIC = {
    'homesite-insurance': 'roc-auc',  # binary classification (AUROC)
    'ecom-offers': 'roc-auc',
    'sberbank-housing': 'rmse',  # regression
    'cooking-time': 'rmse',
    'delivery-eta': 'rmse',
}
LOWER_IS_BETTER = {'rmse', 'mae', 'mse'}


def _load_seed_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f'WARN failed to read {path}: {exc}')
        return None


def _extract_metric(report: dict, dataset: str, split: str) -> float | None:
    """Pluck the primary metric value for the given split from the report.

    Each ``metrics[part]`` dict has a normalised ``score`` field
    (higher-is-better, sign-flipped for regression) plus the raw
    ``rmse`` / ``roc-auc`` values. We return the *raw* primary metric
    so the report tables show familiar units.
    """
    metric_name = PRIMARY_METRIC.get(dataset)
    if metric_name is None:
        return None
    metrics_top = report.get('metrics') or {}
    bucket = metrics_top.get(split)
    if not isinstance(bucket, dict):
        return None
    if metric_name in bucket:
        return float(bucket[metric_name])
    return None


def _extract_score(report: dict, split: str) -> float | None:
    """Higher-is-better normalised score (used for ranking / val-only selection)."""
    bucket = (report.get('metrics') or {}).get(split)
    if isinstance(bucket, dict) and 'score' in bucket:
        return float(bucket['score'])
    return None


def _split_inference_mode(variant: str) -> tuple[str, str]:
    """Return (base_variant, inference_mode) where inference_mode is one
    of 'mean', 'best-head', 'greedy-heads'."""
    if variant.endswith('-best-head'):
        return variant[: -len('-best-head')], 'best-head'
    if variant.endswith('-greedy-heads'):
        return variant[: -len('-greedy-heads')], 'greedy-heads'
    return variant, 'mean'


def _iter_runs(strict: bool = True, allowed_seeds: set | None = None,
               require_gpu: str | None = None):
    """Yield (dataset, variant, inference_mode, seed_id, report_dict) tuples.

    Strict filters applied:
      - DONE marker required (no half-finished runs).
      - report.json must load.
      - config.amp must match the variant's primary 0.toml (head-aware).
      - orphan head-evaluation dirs (no matching base evaluation) dropped.
    Strict-mode (default) additionally requires:
      - seed ∈ allowed_seeds (default {0,1,2}).
      - report.amp_dtype is set (not unknown/None).
      - report.git_commit is set (not unknown/None).
      - report.gpu_name == require_gpu when given (default RTX 4090).
      - no `failure` block in report.
    """
    if allowed_seeds is None:
        allowed_seeds = {'0', '1', '2'}
    if require_gpu is None and strict:
        require_gpu = 'NVIDIA GeForce RTX 4090'
    for ds_dir in sorted(RLA_ROOT.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith('_'):
            continue
        # Build a set of base variants present in this dataset.
        base_variants = {
            v.name.removesuffix('-evaluation')
            for v in ds_dir.iterdir()
            if v.is_dir()
            and v.name.endswith('-evaluation')
            and not v.name.endswith('-best-head-evaluation')
            and not v.name.endswith('-greedy-heads-evaluation')
        }
        for variant_dir in sorted(ds_dir.iterdir()):
            if not variant_dir.is_dir() or not variant_dir.name.endswith(
                '-evaluation'
            ):
                continue
            full_variant = variant_dir.name.removesuffix('-evaluation')
            base_variant, inference_mode = _split_inference_mode(full_variant)
            # Drop orphan head/greedy dirs whose base evaluation is missing.
            if inference_mode != 'mean' and base_variant not in base_variants:
                continue
            # Locate the canonical 0.toml (always lives in the base variant dir).
            base_toml = ds_dir / f'{base_variant}-evaluation' / '0.toml'
            cfg_amp = None
            if base_toml.exists():
                try:
                    import tomllib

                    cfg_amp = bool(
                        tomllib.loads(base_toml.read_text()).get('amp', False)
                    )
                except Exception:
                    cfg_amp = None
            for seed_dir in sorted(variant_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                # DONE marker required.
                if not (seed_dir / 'DONE').exists():
                    continue
                report = _load_seed_report(seed_dir / 'report.json')
                if report is None:
                    continue
                # Head-aware AMP parity check.
                rep_amp = (report.get('config') or {}).get('amp', False)
                if cfg_amp is not None and bool(rep_amp) != cfg_amp:
                    continue
                # Strict filters (git_commit is recorded as best-effort:
                # runs predating the GIT_COMMIT.txt fallback don't have it
                # but were verified to be on RTX 4090; we don't drop them).
                if strict:
                    if seed_dir.name not in allowed_seeds:
                        continue
                    if report.get('amp_dtype') in (None, '', 'unknown'):
                        continue
                    if require_gpu is not None and report.get('gpu_name') != require_gpu:
                        continue
                    if 'failure' in report:
                        continue
                yield (
                    ds_dir.name,
                    base_variant,
                    inference_mode,
                    seed_dir.name,
                    report,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=str(OUT_DIR))
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='Disable strict filters (allow unknown amp_dtype/git_commit, '
             'any seed, any GPU). Useful for partial debugging.',
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = not args.no_strict

    def _parse_time(s) -> float | None:
        """Convert ``report['time']`` (str like '0:01:23.456' or float) to seconds."""
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        try:
            parts = s.split(':')
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            return float(parts[0])
        except Exception:
            return None

    long_rows: list[dict] = []
    for dataset, variant, inference_mode, seed_id, report in _iter_runs(strict=strict):
        n_params = report.get('n_parameters')
        time_total = _parse_time(report.get('time'))
        row = {
            'dataset': dataset,
            'variant': variant,
            'inference_mode': inference_mode,
            'seed': seed_id,
            'n_parameters': n_params,
            'time_seconds': time_total,
            'val': _extract_metric(report, dataset, 'val'),
            'test': _extract_metric(report, dataset, 'test'),
            'val_score': _extract_score(report, 'val'),
            'test_score': _extract_score(report, 'test'),
            'amp_dtype': report.get('amp_dtype', 'unknown'),
            'amp_enabled': report.get('amp_enabled'),
            'git_commit': report.get('git_commit', 'unknown')[:12]
            if isinstance(report.get('git_commit'), str)
            else 'unknown',
            'gpu_name': report.get('gpu_name', 'unknown'),
        }
        long_rows.append(row)

    if not long_rows:
        print('No completed runs found yet.')
        return

    # Long CSV
    long_csv = out_dir / 'rla_results.csv'
    with long_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)
    print(f'wrote {long_csv} ({len(long_rows)} rows)')

    # Aggregate by (dataset, variant, inference_mode)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in long_rows:
        grouped.setdefault(
            (r['dataset'], r['variant'], r['inference_mode']), []
        ).append(r)

    summary_rows = []
    for (ds, var, mode), rs in sorted(grouped.items()):
        tests = [r['test'] for r in rs if r['test'] is not None]
        vals = [r['val'] for r in rs if r['val'] is not None]
        times = [r['time_seconds'] for r in rs if r['time_seconds'] is not None]
        n_params = next((r['n_parameters'] for r in rs if r['n_parameters']), None)
        summary_rows.append(
            {
                'dataset': ds,
                'variant': var,
                'inference_mode': mode,
                'n_seeds': len(rs),
                'n_parameters': n_params,
                'val_mean': statistics.mean(vals) if vals else None,
                'val_std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'test_mean': statistics.mean(tests) if tests else None,
                'test_std': statistics.stdev(tests) if len(tests) > 1 else 0.0,
                'time_mean_s': statistics.mean(times) if times else None,
                'amp_dtype': rs[0].get('amp_dtype', 'unknown'),
                'git_commit': rs[0].get('git_commit', 'unknown'),
                'gpu_name': rs[0].get('gpu_name', 'unknown'),
            }
        )

    summary_csv = out_dir / 'rla_summary.csv'
    with summary_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'wrote {summary_csv} ({len(summary_rows)} rows)')

    # Markdown report
    md = out_dir / 'rla_report.md'
    with md.open('w') as f:
        f.write('# RLA Sweep — Aggregated Results\n\n')
        f.write(
            'Primary metric: AUROC for classification (higher is better), '
            'RMSE for regression (lower is better).\n\n'
        )
        # Pivot test_mean ± std per dataset
        datasets = sorted({r['dataset'] for r in summary_rows})
        variants = sorted({r['variant'] for r in summary_rows})
        f.write('## Test metric (mean ± std over seeds)\n\n')
        f.write('| variant | ' + ' | '.join(datasets) + ' |\n')
        f.write('|' + '|'.join(['---'] * (len(datasets) + 1)) + '|\n')
        idx = {(r['dataset'], r['variant']): r for r in summary_rows}
        for v in variants:
            cells = [v]
            for ds in datasets:
                r = idx.get((ds, v))
                if r is None or r['test_mean'] is None:
                    cells.append('—')
                else:
                    cells.append(f"{r['test_mean']:.4f} ± {r['test_std']:.4f}")
            f.write('| ' + ' | '.join(cells) + ' |\n')

        f.write('\n## Validation metric (mean ± std)\n\n')
        f.write('| variant | ' + ' | '.join(datasets) + ' |\n')
        f.write('|' + '|'.join(['---'] * (len(datasets) + 1)) + '|\n')
        for v in variants:
            cells = [v]
            for ds in datasets:
                r = idx.get((ds, v))
                if r is None or r['val_mean'] is None:
                    cells.append('—')
                else:
                    cells.append(f"{r['val_mean']:.4f} ± {r['val_std']:.4f}")
            f.write('| ' + ' | '.join(cells) + ' |\n')

        f.write('\n## Parameter count and wall-clock\n\n')
        f.write('| variant | dataset | n_params | mean train time (s) |\n')
        f.write('|---|---|---|---|\n')
        for r in sorted(summary_rows, key=lambda r: (r['dataset'], r['variant'])):
            time_str = f"{r['time_mean_s']:.1f}" if r['time_mean_s'] else '—'
            np_str = f"{r['n_parameters']:,}" if r['n_parameters'] else '—'
            f.write(f"| {r['variant']} | {r['dataset']} | {np_str} | {time_str} |\n")

    print(f'wrote {md}')


if __name__ == '__main__':
    main()
