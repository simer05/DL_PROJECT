from __future__ import annotations
import argparse, csv, io, json, os, statistics
from pathlib import Path
import numpy as np
import paramiko
from scipy import stats

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

PAPER_15SEED = {
    'sberbank-housing':  (-0.232078, 0.001217),
    'ecom-offers':       ( 0.590257, 0.001802),
    'homesite-insurance':( 0.962224, 0.000340),
    'cooking-time':      (-0.480366, 0.000155),
    'delivery-eta':      (-0.549912, 0.001687),
}
DATASETS = list(PAPER_15SEED.keys())
HETERO = ['baseline_plr','hetero_raw_lam0.05','hetero_raw_lam0.1','hetero_raw_lam0.2']
CONS   = ['consensus_raw_lam0.05','consensus_raw_lam0.1','consensus_raw_lam0.2']
SOFT   = ['hetero_softmax_lam0.05','hetero_softmax_lam0.1','hetero_softmax_lam0.2']
L1     = ['hetero_l1norm_lam0.05','hetero_l1norm_lam0.1','hetero_l1norm_lam0.2']
HOMO   = [f'homo_{t}_raw_lam{l}' for t in ('xgb','lgbm','cat') for l in (0.05,0.1,0.2)]
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def fetch(sftp, ds, v, n_max):
    rows = []
    for s in range(n_max):
        rp = f'{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/{s}/report.json'
        try:
            with sftp.open(rp) as f: r = json.load(f)
            rows.append({'seed': s,
                         'test': r['metrics']['test']['score'],
                         'val': r['metrics']['val']['score'],
                         'rmse': r['metrics']['test'].get('rmse'),
                         'roc_auc': r['metrics']['test'].get('roc-auc') or r['metrics']['test'].get('roc_auc'),
                         'best_step': r.get('best_step'),
                         'time': r.get('time'),
                         'amp_dtype': r.get('amp_dtype')})
        except FileNotFoundError: pass
    return rows


def msd(xs):
    if not xs: return None, None, 0
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs)


def paired_t(cf_rows, base_rows):
    base_by_seed = {r['seed']: r['test'] for r in base_rows}
    diffs = [r['test'] - base_by_seed[r['seed']] for r in cf_rows if r['seed'] in base_by_seed]
    if len(diffs) < 2: return None, None, len(diffs)
    t, p = stats.ttest_1samp(diffs, 0.0)
    return float(t), float(p), len(diffs)


def boot_ci(diffs, n_boot=2000, seed=0):
    if not diffs: return None, None, None
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path,
                    default=Path('C:/Users/Simerjit Kaur/cf_fisd_recovered/_aggregated'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    print('connecting...')
    gw, tgt = connect(pw); sftp = tgt.open_sftp()

    ALL = HETERO + CONS + SOFT + L1 + HOMO
    long_rows = []
    by_dv = {}
    for ds in DATASETS:
        for v in ALL:
            n_max = 15 if v.startswith(('baseline_plr','hetero_raw')) else 5
            rows = fetch(sftp, ds, v, n_max)
            for r in rows:
                rec = dict(r); rec['dataset']=ds; rec['variant']=v
                long_rows.append(rec)
            by_dv[(ds, v)] = rows
    sftp.close(); tgt.close(); gw.close()

    # Long CSV
    keys = ['dataset','variant','seed','test','val','rmse','roc_auc','best_step','time','amp_dtype']
    with (args.out_dir / 'long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in long_rows: w.writerow({k: r.get(k) for k in keys})
    print(f'wrote long.csv: {len(long_rows)} rows')

    # Wide summary with paired t + bootstrap CI
    summary = []
    for ds in DATASETS:
        base = by_dv.get((ds, 'baseline_plr'), [])
        bm, bs, bn = msd([r['test'] for r in base])
        for v in ALL:
            rows = by_dv.get((ds, v), [])
            if not rows: continue
            m, sd, n = msd([r['test'] for r in rows])
            vm, _, _ = msd([r['val'] for r in rows])
            if v == 'baseline_plr':
                summary.append({'dataset': ds, 'variant': v, 'n': n, 'mean': m, 'std': sd,
                                'val_mean': vm, 'gain': None, 'paired_t': None, 'p': None,
                                'gain_ci_lo': None, 'gain_ci_hi': None})
            else:
                t, p, np_pairs = paired_t(rows, base)
                base_by = {r['seed']: r['test'] for r in base}
                diffs = [r['test'] - base_by[r['seed']] for r in rows if r['seed'] in base_by]
                gm, lo, hi = boot_ci(diffs)
                summary.append({'dataset': ds, 'variant': v, 'n': n, 'mean': m, 'std': sd,
                                'val_mean': vm, 'gain': gm, 'paired_t': t, 'p': p,
                                'gain_ci_lo': lo, 'gain_ci_hi': hi, 'n_pairs': np_pairs})
    with (args.out_dir / 'wide.csv').open('w', newline='') as f:
        keys2 = ['dataset','variant','n','mean','std','val_mean','gain','paired_t','p','gain_ci_lo','gain_ci_hi']
        w = csv.DictWriter(f, fieldnames=keys2, extrasaction='ignore')
        w.writeheader()
        for r in summary: w.writerow({k: r.get(k) for k in keys2})
    print(f'wrote wide.csv: {len(summary)} rows')

    # Markdown headline table
    md = ['# CF-FISD final aggregated results\n', '## Per-(dataset, variant) with paired t-tests\n',
          '| Dataset | Variant | n | mean | std | val_mean | gain | paired_t | p | 95% CI on gain |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        if r['gain'] is None:
            md.append(f"| {r['dataset']} | {r['variant']} | {r['n']} | {r['mean']:+.6f} | {r['std']:.6f} | {r['val_mean']:+.6f} | (ref) | - | - | - |")
        else:
            ci = f"[{r['gain_ci_lo']:+.6f}, {r['gain_ci_hi']:+.6f}]" if r['gain_ci_lo'] is not None else '-'
            md.append(f"| {r['dataset']} | {r['variant']} | {r['n']} | {r['mean']:+.6f} | {r['std']:.6f} | {r['val_mean']:+.6f} | {r['gain']:+.6f} | {r['paired_t']:+.3f} | {r['p']:.4g} | {ci} |")
    md_path = args.out_dir / 'final_table.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f'wrote {md_path}')

    json_path = args.out_dir / 'final_aggregated.json'
    json_path.write_text(json.dumps({'summary': summary, 'long': long_rows[:500]}, indent=2, default=str), encoding='utf-8')
    print(f'wrote {json_path}')


if __name__ == '__main__':
    main()
