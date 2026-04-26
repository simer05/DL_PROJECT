from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
HETERO = ['baseline_plr','hetero_raw_lam0.05','hetero_raw_lam0.1','hetero_raw_lam0.2']
CONS   = ['consensus_raw_lam0.05','consensus_raw_lam0.1','consensus_raw_lam0.2']


def fig1_per_dataset(summary, out_dir: Path):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.5 * len(DATASETS), 4.2), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        rows = [r for r in summary if r['dataset'] == ds and r['n'] >= 5]
        if not rows: continue
        rows.sort(key=lambda r: (r['variant'] != 'baseline_plr', r['variant']))
        labels = [r['variant'].replace('hetero_raw_', 'h_').replace('consensus_raw_', 'c_').replace('hetero_', '').replace('homo_', 'homo-') for r in rows]
        means = [r['mean'] for r in rows]
        stds = [r['std'] for r in rows]
        x = np.arange(len(rows))
        bars = ax.bar(x, means, yerr=stds, capsize=3, edgecolor='black')
        for i, r in enumerate(rows):
            if r['variant'] == 'baseline_plr':
                bars[i].set_color('#888')
            elif r.get('p') is not None and r['p'] < 0.05 and (r['gain'] or 0) > 0:
                bars[i].set_color('#2a9d8f')  # green for significant win
            elif r.get('p') is not None and r['p'] < 0.05 and (r['gain'] or 0) < 0:
                bars[i].set_color('#e76f51')  # red for significant regression
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
        ax.set_title(ds, fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        if rows[0]['variant'] == 'baseline_plr':
            ax.axhline(rows[0]['mean'], color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    fig.suptitle('CF-FISD vs TabM-PLR per dataset (mean ± std). Green = paired-t p<0.05 win; red = significant regression.', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig1_per_dataset.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / 'fig1_per_dataset.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote fig1_per_dataset.png + .pdf')


def fig2_paired_diff(long_csv_path: Path, summary, out_dir: Path):
    rows = []
    with long_csv_path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                r['test'] = float(r['test']); r['seed'] = int(r['seed'])
                rows.append(r)
            except (ValueError, TypeError, KeyError): pass
    by_dv = {}
    for r in rows:
        by_dv.setdefault((r['dataset'], r['variant']), []).append(r)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 3.5))
    for ax, ds in zip(axes, DATASETS):
        # Pick val-best CF-FISD variant for paired-diff hist
        candidates = [r for r in summary if r['dataset'] == ds and r['variant'] != 'baseline_plr' and r['n'] >= 5 and r.get('val_mean') is not None]
        if not candidates:
            ax.set_title(f'{ds}\n(no CF-FISD)', fontsize=9); continue
        best = max(candidates, key=lambda r: r['val_mean'])
        v_best = best['variant']
        cf_rows = by_dv.get((ds, v_best), [])
        base_rows = by_dv.get((ds, 'baseline_plr'), [])
        base_by = {r['seed']: r['test'] for r in base_rows}
        diffs = [r['test'] - base_by[r['seed']] for r in cf_rows if r['seed'] in base_by]
        if not diffs:
            ax.set_title(f'{ds}\n(no pairs)', fontsize=9); continue
        ax.hist(diffs, bins=8, edgecolor='black', alpha=0.75)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(np.mean(diffs), color='green', linestyle='-', linewidth=1.5, label=f'mean Δ = {np.mean(diffs):+.5f}')
        ax.set_title(f'{ds}\n{v_best}', fontsize=9)
        ax.set_xlabel('test_score(CF-FISD) − test_score(baseline)', fontsize=8)
        ax.set_ylabel('seed count', fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
    fig.suptitle('Per-seed paired difference distribution: val-best CF-FISD variant minus TabM-PLR baseline', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_paired_diff.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / 'fig2_paired_diff.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote fig2_paired_diff.png + .pdf')


def fig3_teacher_corr(out_dir: Path):
    DATA = {
        'sberbank-housing':  {'xl': 0.53, 'xc': 0.53, 'lc': 0.64},
        'ecom-offers':       {'xl': 0.32, 'xc': 0.19, 'lc': 0.24},
        'homesite-insurance':{'xl': 0.20, 'xc': 0.49, 'lc': 0.82},
        'cooking-time':      {'xl': -0.02,'xc': 0.37, 'lc': 0.82},
        'delivery-eta':      {'xl': -0.05,'xc': 0.44, 'lc': 0.69},
    }
    teachers = ['XGB','LGBM','CAT']
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.4 * len(DATASETS), 3.6))
    for ax, ds in zip(axes, DATASETS):
        d = DATA[ds]
        rho = np.array([[1.0, d['xl'], d['xc']],
                        [d['xl'], 1.0, d['lc']],
                        [d['xc'], d['lc'], 1.0]])
        im = ax.imshow(rho, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(3)); ax.set_xticklabels(teachers, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(teachers, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{rho[i,j]:.2f}', ha='center', va='center',
                        color='white' if abs(rho[i,j]) > 0.5 else 'black', fontsize=10)
        ax.set_title(ds, fontsize=9)
    fig.suptitle('Teacher pairwise Spearman ρ per dataset', fontsize=10)
    fig.tight_layout()
    fig.colorbar(im, ax=axes, location='right', shrink=0.7)
    fig.savefig(out_dir / 'fig3_teacher_corr.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / 'fig3_teacher_corr.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote fig3_teacher_corr.png + .pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', type=Path,
                    default=Path('C:/Users/Simerjit Kaur/cf_fisd_recovered/_aggregated'))
    args = ap.parse_args()
    args.in_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.in_dir / 'final_aggregated.json'
    if not json_path.exists():
        raise SystemExit(f'{json_path} not found — run final_aggregate.py first')
    state = json.loads(json_path.read_text(encoding='utf-8'))
    summary = state['summary']
    long_csv = args.in_dir / 'long.csv'
    if not long_csv.exists():
        raise SystemExit(f'{long_csv} not found — run final_aggregate.py first')
    fig1_per_dataset(summary, args.in_dir)
    fig2_paired_diff(long_csv, summary, args.in_dir)
    fig3_teacher_corr(args.in_dir)
    print('all figures written under', args.in_dir)


if __name__ == '__main__':
    main()
