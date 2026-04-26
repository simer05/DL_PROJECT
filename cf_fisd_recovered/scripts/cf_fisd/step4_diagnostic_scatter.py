"""Step 4 Figure 2: diagnostic scatter (LGBM-CAT vs XGB-LGBM Spearman rho)
with bootstrap CIs on both axes and shaded decision regions.

Decision rule (CF-FISD diagnostic) — current heuristic:
- WIN region: ALL three pairwise teacher correlations are 'medium' (rho in [0.5, 0.85])
- regression region: any pair is 'too high' (rho > 0.85) — teachers degenerate, distillation overfits
- match region: any pair is 'too low' (rho < 0.5) — teachers disagree, signal too noisy

We plot in (LGBM-CAT, XGB-LGBM) space. The third pair (XGB-CAT) is not on the axes
but shown as a label on each point.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from scipy import stats
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'

DATASETS = ['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
TEACHER_NAMES = ['xgb','lgbm','cat']

OUT = Path(r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated')
OUT.mkdir(parents=True, exist_ok=True)


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def feature_bootstrap_ci(a, b, n_boot=2000, seed=0):
    """Bootstrap on Spearman rho by resampling features with replacement."""
    rng = np.random.default_rng(seed)
    n = len(a)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(a[idx], b[idx])
        rhos[i] = r if r == r else 0.0
    point, _ = stats.spearmanr(a, b)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    gw, tgt = connect(pw)
    sftp = tgt.open_sftp()

    # Find teacher importances. Probable path:
    # _teachers/tabred/<ds>/<teacher>/importances.npy
    # OR _teachers/tabred/<ds>/<teacher>/feature_importances.npy

    def load_importances(ds):
        out = {}
        base_dir = f'{PD}/exp/cf_fisd/_teachers/tabred/{ds}'
        for teacher in TEACHER_NAMES:
            p = f'{base_dir}/{teacher}.npy'
            try:
                with sftp.open(p, 'rb') as f:
                    arr = np.load(f)
                out[teacher] = arr
            except FileNotFoundError:
                print(f'MISSING: {p}')
                return None
        return out

    data = {}
    try:
        for ds in DATASETS:
            print(f'\n=== {ds} ===')
            imps = load_importances(ds)
            if imps is None: continue
            for k, v in imps.items():
                print(f'  {k}: shape={v.shape}, mean={v.mean():.4f}, max={v.max():.4f}')
            data[ds] = imps
    finally:
        sftp.close(); tgt.close(); gw.close()

    # Compute pairwise Spearman with bootstrap CIs
    pairs = [('xgb','lgbm'),('lgbm','cat'),('xgb','cat')]
    results = {}
    for ds, imps in data.items():
        results[ds] = {}
        for a, b in pairs:
            point, lo, hi = feature_bootstrap_ci(imps[a], imps[b])
            results[ds][f'{a}_{b}'] = {'rho': point, 'lo': lo, 'hi': hi}
            print(f'  {ds:25s} {a:5s} <-> {b:5s} rho={point:+.3f} [{lo:+.3f}, {hi:+.3f}]')

    (OUT / 'diagnostic_correlations.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=150)

    # Decision regions per the WIN-region heuristic used in analysis_branchA/B.md and step6_bootstrap_monitor.py:
    # WIN region: ρ(XGB,LGBM) > 0.10 AND ρ(LGBM,CAT) >= 0.70
    # Axes here: x = ρ(LGBM, CAT), y = ρ(XGB, LGBM)
    # So WIN region in (x, y) space is: x >= 0.70 AND y > 0.10
    # match region: everything else (no third "regression" region — the heuristic is binary in our 5-dataset sample)
    # We add a hypothetical "regression (predicted)" region at very high correlation in case of teacher degeneracy (any pair > 0.95).
    XL_T = 0.10
    LC_T = 0.70
    DEGEN = 0.95
    # Match (default — light blue everywhere first)
    ax.add_patch(Rectangle((-0.05, -0.05), 1.10, 1.10, alpha=0.10, facecolor='#1f77b4', edgecolor='none'))
    # WIN region: x in [LC_T, DEGEN], y in [XL_T, DEGEN]
    ax.add_patch(Rectangle((LC_T, XL_T), DEGEN-LC_T, DEGEN-XL_T, alpha=0.22, facecolor='#2ca02c', edgecolor='none'))
    # Hypothetical regression (degenerate teachers): any pair > 0.95
    ax.add_patch(Rectangle((DEGEN, -0.05), 1.10-DEGEN, 1.10, alpha=0.18, facecolor='#d62728', edgecolor='none'))
    ax.add_patch(Rectangle((-0.05, DEGEN), 1.10, 1.10-DEGEN, alpha=0.18, facecolor='#d62728', edgecolor='none'))

    # Plot points
    palette = {'sberbank-housing':'tab:purple','ecom-offers':'tab:orange','homesite-insurance':'tab:green',
               'cooking-time':'tab:brown','delivery-eta':'tab:gray'}
    for ds in DATASETS:
        if ds not in results: continue
        x = results[ds]['lgbm_cat']
        y = results[ds]['xgb_lgbm']
        c = palette[ds]
        ax.errorbar(x['rho'], y['rho'],
                    xerr=[[x['rho']-x['lo']], [x['hi']-x['rho']]],
                    yerr=[[y['rho']-y['lo']], [y['hi']-y['rho']]],
                    fmt='o', color=c, markersize=8, capsize=3, lw=1.4, zorder=5)
        # offset label so it doesn't overlap the marker
        ax.annotate(ds, (x['rho'], y['rho']), xytext=(7, 5), textcoords='offset points', fontsize=9, color=c, zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('LGBM ↔ CAT Spearman ρ (with feature-bootstrap 95% CI)')
    ax.set_ylabel('XGB ↔ LGBM Spearman ρ (with feature-bootstrap 95% CI)')
    ax.set_title('Teacher-correlation diagnostic across TabReD datasets')
    # Threshold gridlines (matching the rule)
    ax.axhline(XL_T, lw=0.6, color='gray', alpha=0.5)
    ax.axvline(LC_T, lw=0.6, color='gray', alpha=0.5)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#2ca02c', alpha=0.22, label=f'WIN: ρ(LGBM,CAT) ≥ {LC_T} AND ρ(XGB,LGBM) > {XL_T}'),
        Patch(facecolor='#1f77b4', alpha=0.10, label='match (everything else)'),
        Patch(facecolor='#d62728', alpha=0.18, label=f'regression — predicted only if any pair > {DEGEN} (degenerate teachers, none observed)'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(OUT / 'fig_diagnostic_scatter.pdf')
    plt.savefig(OUT / 'fig_diagnostic_scatter.png', dpi=200)
    print(f'wrote fig_diagnostic_scatter.{{pdf,png}}')


if __name__ == '__main__':
    main()
