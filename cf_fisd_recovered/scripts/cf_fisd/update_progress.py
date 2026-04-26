from __future__ import annotations
import datetime, io, json, os, statistics
from pathlib import Path
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PAPER = {
    'sberbank-housing':  (-0.232078, 0.001217),
    'ecom-offers':       ( 0.590257, 0.001802),
    'homesite-insurance':( 0.962224, 0.000340),
    'cooking-time':      (-0.480366, 0.000155),
    'delivery-eta':      (-0.549912, 0.001687),
}
DATASETS = list(PAPER.keys())
HETERO = ['baseline_plr','hetero_raw_lam0.05','hetero_raw_lam0.1','hetero_raw_lam0.2']
CONS = ['consensus_raw_lam0.05','consensus_raw_lam0.1','consensus_raw_lam0.2']
SOFT = ['hetero_softmax_lam0.05','hetero_softmax_lam0.1','hetero_softmax_lam0.2']
L1   = ['hetero_l1norm_lam0.05','hetero_l1norm_lam0.1','hetero_l1norm_lam0.2']
HOMO = [f'homo_{t}_raw_lam{l}' for t in ('xgb','lgbm','cat') for l in (0.05,0.1,0.2)]
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'
OUT = Path('C:/Users/Simerjit Kaur/cf_fisd_recovered/RESULTS_AND_PROGRESS.md')

def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt

def msd(rows, k='test'):
    xs=[r[k] for r in rows]
    if not xs: return None,None,0
    return statistics.mean(xs), statistics.stdev(xs) if len(xs)>1 else 0.0, len(xs)

def fetch(sftp, ds, v, n_max):
    rows=[]
    for s in range(n_max):
        try:
            with sftp.open(f'{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/{s}/report.json') as f: r=json.load(f)
            rows.append({'seed':s,'test':r['metrics']['test']['score'],'val':r['metrics']['val']['score']})
        except FileNotFoundError: pass
    return rows

def main():
    pw=os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    gw, tgt = connect(pw); sftp=tgt.open_sftp()
    data={ds:{} for ds in DATASETS}
    ALL = HETERO + CONS + SOFT + L1 + HOMO
    for ds in DATASETS:
        for v in ALL:
            data[ds][v] = fetch(sftp, ds, v, 15 if v in HETERO else 5)
    _, o, _ = tgt.exec_command('qstat -u simerjit 2>&1 | awk "NR>5" | wc -l', timeout=30)
    qcount = int(o.read().decode().strip() or 0)
    sftp.close(); tgt.close(); gw.close()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [f'# CF-FISD Live Results\n_Auto-updated {now}_\n', '## Per-(dataset, variant) numbers from cluster\n',
             '| Dataset | Variant | n | test_mean | test_std | val_mean | gain | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        bm, bs, bn = msd(data[ds]['baseline_plr'])
        bvm, _, _ = msd(data[ds]['baseline_plr'], 'val') if data[ds]['baseline_plr'] else (None, None, None)
        if bn: lines.append(f'| {ds} | baseline_plr | {bn} | {bm:+.6f} | {bs:.6f} | {bvm:+.6f} | (ref) | - |')
        for v in ALL:
            if v == 'baseline_plr': continue
            m, sd, n = msd(data[ds][v])
            if n == 0: continue
            vm, _, _ = msd(data[ds][v], 'val')
            gain = m - bm if bn else None
            verdict = '-'
            if bn and bs is not None and bs > 0:
                verdict = ('WIN' if gain > bs and n >= 15 else
                           'promising' if gain > bs and n < 15 else
                           'matches' if abs(gain) <= bs else
                           'WORSE' if n >= 15 else 'partial')
            lines.append(f'| {ds} | {v} | {n} | {m:+.6f} | {sd:.6f} | {vm:+.6f} | {gain:+.6f} | {verdict} |')
    lines.append(f'\n## Queue: **{qcount}** in NSCC queue right now')
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {OUT}')
    print(f'total cells: {sum(len(v) for ds in DATASETS for v in data[ds].values())}')

if __name__ == '__main__': main()
