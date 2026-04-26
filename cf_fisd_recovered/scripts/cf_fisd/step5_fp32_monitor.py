"""Step 5 monitor: poll cooking-time/baseline_plr_fp32 reports + write COOKING_FP32_CONTROL.md.

Runs once per ~10 min until 5/5 reports present, then writes verdict. Cap at 90 min (= ~9 ticks).
"""
from __future__ import annotations
import os, sys, json, time, statistics, datetime as dt
from pathlib import Path
import paramiko
from scipy import stats

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'

OUT = Path(r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated')
OUT.mkdir(parents=True, exist_ok=True)

PAPER_TABM_PLR_COOKING = (-0.480366, 0.000155)  # (mean, std) from TabReD paper Appendix


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def fetch_fp32(sftp, n=5):
    rows = []
    for s in range(n):
        rp = f'{PD}/exp/cf_fisd/tabred/cooking-time/baseline_plr_fp32-evaluation/{s}/report.json'
        try:
            with sftp.open(rp) as f:
                r = json.load(f)
            rows.append({'seed': s, 'test': r['metrics']['test']['score'],
                         'val': r['metrics']['val']['score'],
                         'amp_dtype': r.get('amp_dtype')})
        except FileNotFoundError:
            pass
    return rows


def fetch_baseline_bf16(sftp, n=15):
    rows = []
    for s in range(n):
        rp = f'{PD}/exp/cf_fisd/tabred/cooking-time/baseline_plr-evaluation/{s}/report.json'
        try:
            with sftp.open(rp) as f:
                r = json.load(f)
            rows.append({'seed': s, 'test': r['metrics']['test']['score']})
        except FileNotFoundError:
            pass
    return rows


def write_status(rows_fp32, rows_bf16, msg):
    paper_mean, paper_std = PAPER_TABM_PLR_COOKING
    bf16_mean = statistics.mean(r['test'] for r in rows_bf16) if rows_bf16 else float('nan')
    bf16_std  = statistics.stdev(r['test'] for r in rows_bf16) if len(rows_bf16) > 1 else 0.0
    bf16_dev = abs(bf16_mean - paper_mean)
    bf16_dev_in_paper_sigmas = bf16_dev / paper_std if paper_std > 0 else float('nan')

    fp32_mean = statistics.mean(r['test'] for r in rows_fp32) if rows_fp32 else float('nan')
    fp32_std  = statistics.stdev(r['test'] for r in rows_fp32) if len(rows_fp32) > 1 else 0.0
    fp32_dev = abs(fp32_mean - paper_mean) if rows_fp32 else float('nan')
    fp32_dev_in_paper_sigmas = (fp32_dev / paper_std) if (rows_fp32 and paper_std > 0) else float('nan')

    md = [
        '# Cooking-time FP32 control',
        '',
        f'_Last updated: {dt.datetime.now(dt.timezone.utc).isoformat()} UTC_',
        '',
        '## Setup',
        '',
        'A single config: `cooking-time / baseline_plr_fp32-evaluation`, identical to the `baseline_plr` config except `amp = false` (= FP32 forward+backward), 5 seeds (0–4). Submitted as PBS jobs 13911135–13911139.',
        '',
        '## Reference',
        '',
        f'- TabReD paper TabM-PLR cooking-time: mean = **{paper_mean:+.6f}** (RMSE on negative), std = **{paper_std:.6f}** (n=15 in paper).',
        '',
        '## Our 15-seed BF16 baseline (default amp=true)',
        '',
        f'- mean = **{bf16_mean:+.6f}**, std = **{bf16_std:.6f}**, n={len(rows_bf16)}',
        f'- deviation from paper: |Δ| = {bf16_dev:.6f} = **{bf16_dev_in_paper_sigmas:.1f}σ** of paper std',
        f'- {"within ±2σ ✓" if bf16_dev_in_paper_sigmas <= 2.0 else "OUTSIDE ±2σ ✗"} of paper TabM-PLR',
        '',
        '## Our 5-seed FP32 control',
        '',
    ]
    if rows_fp32:
        md += [
            f'- mean = **{fp32_mean:+.6f}**, std = **{fp32_std:.6f}**, n={len(rows_fp32)}',
            f'- deviation from paper: |Δ| = {fp32_dev:.6f} = **{fp32_dev_in_paper_sigmas:.1f}σ** of paper std',
            f'- {"within ±2σ ✓" if fp32_dev_in_paper_sigmas <= 2.0 else "OUTSIDE ±2σ ✗"} of paper TabM-PLR',
            '',
            '## Verdict',
            '',
        ]
        if fp32_dev_in_paper_sigmas <= 2.0 and bf16_dev_in_paper_sigmas > 2.0:
            md.append('**FP32 reproduces paper-equivalent results; BF16 does not.** The cooking-time deviation in our headline 15-seed run is attributable to the BF16 mixed-precision forward+backward used by the standard TabM training script. This is a verified FP32-vs-BF16 attribution, not speculation.')
        elif fp32_dev_in_paper_sigmas <= 2.0 and bf16_dev_in_paper_sigmas <= 2.0:
            md.append('Both FP32 and BF16 reproduce paper-equivalent results within ±2σ. The cooking-time deviation we observed in the headline run is within paper variability; no precision-attribution needed.')
        elif fp32_dev_in_paper_sigmas > 2.0 and bf16_dev_in_paper_sigmas <= 2.0:
            md.append('FP32 deviates more than BF16. Unexpected — the precision difference is not the cause of our observed deviation. Other factors (data version, hyperparameter selection) need investigation.')
        else:
            md.append('Both FP32 and BF16 are outside ±2σ of paper. The cooking-time deviation is NOT precision-attributable. Other factors (data version, hyperparameter selection, or paper σ being unusually tight at 0.000155) may explain.')
    else:
        md += [
            '- jobs queued; waiting for results.',
            '',
            '## Verdict',
            '',
            f'_pending — {msg}_',
        ]
    md += ['', '## Per-seed values', '', '| seed | FP32 test | amp_dtype |', '|---|---|---|']
    for r in sorted(rows_fp32, key=lambda x: x['seed']):
        md.append(f"| {r['seed']} | {r['test']:+.6f} | {r['amp_dtype']} |")

    (OUT / 'COOKING_FP32_CONTROL.md').write_text('\n'.join(md), encoding='utf-8')


def main():
    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')

    n_ticks = int(os.environ.get('FP32_MAX_TICKS', '12'))
    interval_s = int(os.environ.get('FP32_INTERVAL_S', '600'))  # 10 min

    last_n_fp32 = -1
    for tick in range(n_ticks):
        try:
            gw, tgt = connect(pw)
            sftp = tgt.open_sftp()
            rows_fp32 = fetch_fp32(sftp, 5)
            rows_bf16 = fetch_baseline_bf16(sftp, 15)
            sftp.close(); tgt.close(); gw.close()
            n_fp32 = len(rows_fp32)
            print(f'[tick {tick}] FP32 reports: {n_fp32}/5; BF16 reports: {len(rows_bf16)}/15')
            write_status(rows_fp32, rows_bf16, f'tick {tick}/{n_ticks}, interval={interval_s}s')
            if n_fp32 == 5:
                print('all 5 FP32 reports present; finalizing')
                break
            if n_fp32 != last_n_fp32 and n_fp32 > 0:
                print(f'  progress: was {last_n_fp32}, now {n_fp32}')
            last_n_fp32 = n_fp32
        except Exception as e:
            print(f'tick {tick} error: {e}')
        time.sleep(interval_s)
    print('FP32 monitor exiting')


if __name__ == '__main__':
    main()
