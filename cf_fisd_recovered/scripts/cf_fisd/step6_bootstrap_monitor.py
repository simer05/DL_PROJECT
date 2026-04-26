"""Step 6: teacher-seed bootstrap monitor.

Each iteration:
1. SSH to cluster, count completed (DS, SEED) cells (all of xgb.npy/lgbm.npy/cat.npy + meta.json present).
2. For each completed cell, load importances and compute Spearman rho.
3. Aggregate over completed seeds per dataset → bootstrap CI on each pair.
4. Re-check the diagnostic rule under bootstrap CIs.
5. Write BOOTSTRAP_STATUS.md.
6. If not 80% complete by HARD_CUTOFF (UTC ISO timestamp), append a recommendation line.

Run once per ~30 min; keep alive until cutoff.
"""
from __future__ import annotations
import os, sys, json, time, datetime as dt, subprocess
from pathlib import Path
import numpy as np
from scipy import stats
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'
DATASETS = ['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
SEEDS = [1, 2, 3, 4]  # bootstrap seeds (seed=0 is the canonical "original" teacher set in _teachers/tabred/<ds>/)
TEACHERS = ['xgb','lgbm','cat']
TOTAL_CELLS = len(DATASETS) * len(SEEDS)  # 20

# Decision rule (WIN region heuristic from main analysis):
RHO_XL_THRESHOLD = 0.10  # ρ(XGB, LGBM) > 0.10
RHO_LC_THRESHOLD = 0.70  # ρ(LGBM, CAT) ≥ 0.70

# Actual CF-FISD outcome per dataset (from headline 15-seed paired tests, post-Bonferroni):
ACTUAL_OUTCOME = {
    'homesite-insurance': 'WIN',
    'cooking-time': 'match',
    'delivery-eta': 'match',
    'sberbank-housing': 'match',
    'ecom-offers': 'match',
}

OUT = Path(r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated')
OUT.mkdir(parents=True, exist_ok=True)


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def cell_complete(sftp, seed, ds):
    base = f'{PD}/exp/cf_fisd/_teachers/tabred_seed{seed}/{ds}'
    for fname in ('xgb.npy','lgbm.npy','cat.npy','meta.json'):
        try:
            sftp.stat(f'{base}/{fname}')
        except FileNotFoundError:
            return False
    return True


def load_cell(sftp, seed, ds):
    base = f'{PD}/exp/cf_fisd/_teachers/tabred_seed{seed}/{ds}'
    out = {}
    for t in TEACHERS:
        with sftp.open(f'{base}/{t}.npy', 'rb') as f:
            out[t] = np.load(f)
    return out


def spearman(a, b):
    r, _ = stats.spearmanr(a, b)
    return float(r) if r == r else 0.0


def classify(xl, lc):
    return 'WIN' if (xl > RHO_XL_THRESHOLD and lc >= RHO_LC_THRESHOLD) else 'match'


def run_once(pw, hard_cutoff_utc):
    gw, tgt = connect(pw)
    sftp = tgt.open_sftp()
    try:
        completed = {}  # (seed, ds) -> imps
        for s in SEEDS:
            for ds in DATASETS:
                if cell_complete(sftp, s, ds):
                    completed[(s, ds)] = load_cell(sftp, s, ds)

        # Get qstat for in-progress
        _, so, _ = tgt.exec_command('qstat -u simerjit')
        qs = so.read().decode('utf-8','replace')
    finally:
        sftp.close(); tgt.close(); gw.close()

    # Aggregate per dataset → bootstrap CI on each pair
    boot_results = {}
    for ds in DATASETS:
        seeds_done_for_ds = [s for s in SEEDS if (s, ds) in completed]
        if not seeds_done_for_ds:
            boot_results[ds] = {'n_seeds': 0}
            continue
        # for each pair, compute rho per seed
        per_seed = {pair: [] for pair in (('xgb','lgbm'),('lgbm','cat'),('xgb','cat'))}
        for s in seeds_done_for_ds:
            imps = completed[(s, ds)]
            for a, b in per_seed:
                per_seed[(a,b)].append(spearman(imps[a], imps[b]))
        # Stats per pair
        ds_summary = {'n_seeds': len(seeds_done_for_ds), 'seeds': seeds_done_for_ds}
        for pair, vals in per_seed.items():
            arr = np.asarray(vals)
            ds_summary[f'{pair[0]}_{pair[1]}'] = {
                'mean': float(arr.mean()),
                'lo': float(arr.min()),
                'hi': float(arr.max()),
                'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                'values': vals,
            }
        # Classify under bootstrap
        xl_lo = ds_summary['xgb_lgbm']['lo']
        xl_hi = ds_summary['xgb_lgbm']['hi']
        lc_lo = ds_summary['lgbm_cat']['lo']
        lc_hi = ds_summary['lgbm_cat']['hi']
        # Bootstrap robust: rule prediction must be the same for *all* corner combinations
        all_predictions = set()
        for xl in (xl_lo, xl_hi, ds_summary['xgb_lgbm']['mean']):
            for lc in (lc_lo, lc_hi, ds_summary['lgbm_cat']['mean']):
                all_predictions.add(classify(xl, lc))
        ds_summary['rule_predictions_under_bootstrap'] = sorted(all_predictions)
        ds_summary['rule_robust'] = (len(all_predictions) == 1)
        ds_summary['rule_predicted'] = classify(arr.mean() if False else ds_summary['xgb_lgbm']['mean'], ds_summary['lgbm_cat']['mean'])
        ds_summary['actual_outcome'] = ACTUAL_OUTCOME[ds]
        ds_summary['classified_correctly'] = (ds_summary['rule_predicted'] == ACTUAL_OUTCOME[ds])
        boot_results[ds] = ds_summary

    # Write status
    n_complete = len(completed)
    pct = 100.0 * n_complete / TOTAL_CELLS
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_sgt = now_utc + dt.timedelta(hours=8)
    md = [
        '# Teacher-seed bootstrap status',
        '',
        f'- **Last update**: {now_utc.isoformat()} UTC = {now_sgt.strftime("%Y-%m-%d %H:%M")} SGT',
        f'- **Hard cutoff**: {hard_cutoff_utc.isoformat()} UTC = {(hard_cutoff_utc + dt.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")} SGT',
        f'- **Cells complete**: {n_complete}/{TOTAL_CELLS} = **{pct:.0f}%**',
        '',
        '## Per-cell completion (5 datasets × 4 seeds = 20 cells)',
        '',
        '|  | seed=1 | seed=2 | seed=3 | seed=4 |',
        '|---|---|---|---|---|',
    ]
    for ds in DATASETS:
        row = [f'**{ds}**']
        for s in SEEDS:
            row.append('✓' if (s, ds) in completed else '·')
        md.append('| ' + ' | '.join(row) + ' |')

    md += ['', '## Bootstrap rho summary per dataset', '',
           '| Dataset | n seeds done | ρ(XGB,LGBM) range | ρ(LGBM,CAT) range | Rule prediction | Bootstrap-robust? | Actual outcome | Correct? |',
           '|---|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        b = boot_results[ds]
        if b['n_seeds'] == 0:
            md.append(f"| {ds} | 0 | - | - | - | - | {ACTUAL_OUTCOME[ds]} | - |")
            continue
        md.append(f"| {ds} | {b['n_seeds']} | "
                  f"[{b['xgb_lgbm']['lo']:+.3f}, {b['xgb_lgbm']['hi']:+.3f}] | "
                  f"[{b['lgbm_cat']['lo']:+.3f}, {b['lgbm_cat']['hi']:+.3f}] | "
                  f"{b['rule_predicted']} | "
                  f"{'YES' if b['rule_robust'] else 'NO'} | "
                  f"{b['actual_outcome']} | "
                  f"{'YES' if b['classified_correctly'] else 'NO'} |")

    md += ['', '## Recommendation',]
    rule_robust_overall = all(boot_results[ds].get('rule_robust', False) for ds in DATASETS if boot_results[ds]['n_seeds'] > 0)
    n_with_data = sum(1 for ds in DATASETS if boot_results[ds]['n_seeds'] > 0)
    all_with_data_correct = all(boot_results[ds].get('classified_correctly', False) for ds in DATASETS if boot_results[ds]['n_seeds'] > 0)
    full_coverage = (n_with_data == len(DATASETS))
    branch_a_ready = full_coverage and all_with_data_correct  # mean-classification correct on all 5

    past_cutoff = now_utc >= hard_cutoff_utc
    over_80 = pct >= 80.0
    if past_cutoff and not over_80:
        md.append('')
        md.append(f'**HARD CUTOFF REACHED — bootstrap not at 80%** (current: {pct:.0f}%, datasets with data: {n_with_data}/{len(DATASETS)}).')
        if branch_a_ready:
            md.append('Partial bootstrap classifies all 5 datasets correctly at the mean. **Submit Branch A (validated predictor framing) but clearly note bootstrap is partial.**')
        else:
            md.append('Partial bootstrap is missing dataset(s) needed to validate rule (esp. homesite for the WIN classification). **Submit Branch B (descriptive heuristic framing).**')
    elif over_80 and branch_a_ready:
        md.append('')
        md.append('Bootstrap ≥80% complete; rule classifies all 5 datasets correctly under bootstrap means. ' +
                  ('All bootstrap CIs are rule-robust — strongest evidence. ' if rule_robust_overall else
                   'Some CIs cross thresholds (boundary cases). ') +
                  '**Submit Branch A (validated predictor framing).**')
    elif over_80:
        md.append('')
        md.append('Bootstrap ≥80% complete but mean classification is INCORRECT for at least one dataset. **Submit Branch B (descriptive heuristic framing).**')
    else:
        eta_hours = (hard_cutoff_utc - now_utc).total_seconds() / 3600
        md.append('')
        md.append(f'In progress. {pct:.0f}% complete, {eta_hours:+.1f} hours until hard cutoff. Re-evaluating every 30 min.')

    md.append('')
    md.append('## Raw JSON')
    md.append('')
    md.append('```json')
    md.append(json.dumps({'completion_pct': pct, 'cells_done': n_complete, 'cells_total': TOTAL_CELLS,
                          'hard_cutoff_utc': hard_cutoff_utc.isoformat(),
                          'last_update_utc': now_utc.isoformat(),
                          'rule_robust_overall': rule_robust_overall,
                          'all_classified_correct_at_mean': all_with_data_correct,
                          'full_coverage': full_coverage,
                          'branch_a_ready': branch_a_ready,
                          'per_dataset': {ds: {k: v for k, v in boot_results[ds].items()
                                               if k != 'values'} for ds in DATASETS},
                          'qstat': qs}, indent=2, default=str))
    md.append('```')

    (OUT / 'BOOTSTRAP_STATUS.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'[{now_utc.isoformat()}] {n_complete}/{TOTAL_CELLS} = {pct:.0f}% complete; rule_robust={rule_robust_overall}; correct_at_mean={all_with_data_correct}; branch_a_ready={branch_a_ready}')
    return n_complete, pct, rule_robust_overall, all_with_data_correct


def main():
    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    # Hard cutoff: 4 hours from script start (= ~19:00 SGT today if started ~15:00 SGT)
    cutoff_str = os.environ.get('CUTOFF_UTC')
    if cutoff_str:
        cutoff = dt.datetime.fromisoformat(cutoff_str.replace('Z','+00:00'))
    else:
        cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)
    print(f'Hard cutoff: {cutoff.isoformat()} UTC')

    one_shot = '--once' in sys.argv
    if one_shot:
        run_once(pw, cutoff)
        return

    while True:
        try:
            run_once(pw, cutoff)
        except Exception as e:
            print(f'tick error: {e}')
        if dt.datetime.now(dt.timezone.utc) >= cutoff:
            print('cutoff reached; one final tick written; exiting')
            break
        time.sleep(30 * 60)


if __name__ == '__main__':
    main()
