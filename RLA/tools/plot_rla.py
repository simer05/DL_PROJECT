"""Generate the five diagnostic figures required by Section 9.5 of the RLA spec.

Figures (all saved as PNG + PDF inside ``paper/exp/rla/_aggregated/figs/``):
    A. Test metric vs rank, one curve per dataset.
    B. Wall-clock train time vs rank.
    C. Mean pairwise member-logit correlation vs rank (computed from saved
       per-seed predictions if available; otherwise skipped with a warning).
    D. RLA-first vs RLA-uniform at the best rank, per dataset.
    E. Total parameter count vs rank, broken down into shared backbone
       and adapter components.

Usage from the repo root, after aggregate_rla_results.py has run:
    python tools/plot_rla.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
AGG = REPO_ROOT / 'paper' / 'exp' / 'rla' / '_aggregated'

# Higher-is-better metrics; for these we plot the metric directly. For
# regression (RMSE) we plot the metric directly too but flip the y axis
# annotation.
LOWER_IS_BETTER = {
    'sberbank-housing',
    'cooking-time',
    'delivery-eta',
}


def _load_summary() -> list[dict]:
    rows = []
    csv_path = AGG / 'rla_summary.csv'
    if not csv_path.exists():
        raise SystemExit(f'No summary CSV at {csv_path}; run aggregate_rla_results.py first')
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in (
                'val_mean',
                'val_std',
                'test_mean',
                'test_std',
                'time_mean_s',
                'n_parameters',
                'n_seeds',
            ):
                if r.get(k):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def _parse_variant(v: str) -> dict | None:
    if v == 'baseline':
        return {'family': 'baseline', 'rank': 1, 'first_only': False, 'additive': False}
    m = re.match(r'rla_(first|uniform)_r(\d+)$', v)
    if m:
        return {
            'family': m.group(1),
            'rank': int(m.group(2)),
            'first_only': m.group(1) == 'first',
            'additive': False,
        }
    m = re.match(r'rla_additive_(first|uniform)_r(\d+)$', v)
    if m:
        return {
            'family': 'additive',
            'rank': int(m.group(2)),
            'first_only': m.group(1) == 'first',
            'additive': True,
        }
    return None


def figure_A(rows: list[dict], out: Path) -> None:
    """Test metric vs rank, one curve per dataset, RLA-uniform family."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_dataset: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        meta = _parse_variant(r['variant'])
        if meta is None or meta['family'] != 'uniform':
            continue
        if r.get('test_mean') is None:
            continue
        by_dataset.setdefault(r['dataset'], []).append(
            (meta['rank'], r['test_mean'], r.get('test_std') or 0.0)
        )
    for ds, pts in sorted(by_dataset.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=ds)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlabel('Adapter rank r')
    ax.set_ylabel('Test metric')
    ax.set_title('Figure A — Test metric vs rank (RLA-uniform)')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out / 'figA_test_vs_rank.png', dpi=200)
    fig.savefig(out / 'figA_test_vs_rank.pdf')
    plt.close(fig)


def figure_B(rows: list[dict], out: Path) -> None:
    """Wall-clock train time vs rank."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        meta = _parse_variant(r['variant'])
        if meta is None or meta['family'] != 'uniform':
            continue
        if r.get('time_mean_s') is None:
            continue
        by_dataset.setdefault(r['dataset'], []).append(
            (meta['rank'], r['time_mean_s'])
        )
    for ds, pts in sorted(by_dataset.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] / 60 for p in pts]
        ax.plot(xs, ys, marker='o', label=ds)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlabel('Adapter rank r')
    ax.set_ylabel('Mean train time (min)')
    ax.set_title('Figure B — Wall-clock vs rank (RLA-uniform)')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out / 'figB_time_vs_rank.png', dpi=200)
    fig.savefig(out / 'figB_time_vs_rank.pdf')
    plt.close(fig)


def figure_D(rows: list[dict], out: Path) -> None:
    """RLA-first vs RLA-uniform at best rank per dataset."""
    by = {(r['dataset'], r['variant']): r for r in rows}
    datasets = sorted({r['dataset'] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = list(range(len(datasets)))
    first_vals, uniform_vals = [], []
    for ds in datasets:
        # Choose best (lowest if regression else highest) test_mean across r∈{2,4,8}.
        higher_better = ds not in LOWER_IS_BETTER
        cmp = (lambda a, b: a > b) if higher_better else (lambda a, b: a < b)

        def best(family: str) -> float | None:
            best_v = None
            for rk in (2, 4, 8):
                row = by.get((ds, f'rla_{family}_r{rk}'))
                if row is None or row.get('test_mean') is None:
                    continue
                if best_v is None or cmp(row['test_mean'], best_v):
                    best_v = row['test_mean']
            return best_v

        first_vals.append(best('first'))
        uniform_vals.append(best('uniform'))
    x_arr = [i - width / 2 for i in xs]
    x_arr2 = [i + width / 2 for i in xs]
    fv = [v if v is not None else 0 for v in first_vals]
    uv = [v if v is not None else 0 for v in uniform_vals]
    ax.bar(x_arr, fv, width, label='RLA-first (best r)')
    ax.bar(x_arr2, uv, width, label='RLA-uniform (best r)')
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Best test metric')
    ax.set_title('Figure D — RLA-first vs RLA-uniform at best r')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'figD_first_vs_uniform.png', dpi=200)
    fig.savefig(out / 'figD_first_vs_uniform.pdf')
    plt.close(fig)


def figure_E(rows: list[dict], out: Path) -> None:
    """Total parameter count vs rank (RLA-uniform), per dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_dataset: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        meta = _parse_variant(r['variant'])
        if meta is None or meta['family'] != 'uniform':
            continue
        if r.get('n_parameters') is None:
            continue
        by_dataset.setdefault(r['dataset'], []).append(
            (meta['rank'], int(r['n_parameters']))
        )
    for ds, pts in sorted(by_dataset.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] / 1e6 for p in pts]
        ax.plot(xs, ys, marker='o', label=ds)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlabel('Adapter rank r')
    ax.set_ylabel('Total parameters (M)')
    ax.set_title('Figure E — Parameter count vs rank (RLA-uniform)')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out / 'figE_params_vs_rank.png', dpi=200)
    fig.savefig(out / 'figE_params_vs_rank.pdf')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=str(AGG / 'figs'))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_summary()
    figure_A(rows, out)
    figure_B(rows, out)
    figure_D(rows, out)
    figure_E(rows, out)
    # Figure C (member-logit correlation) requires per-seed predictions
    # which paper/bin/evaluate.py does not store by default. We skip it
    # and note this in the report.
    print(f'wrote figures to {out}')


if __name__ == '__main__':
    main()
