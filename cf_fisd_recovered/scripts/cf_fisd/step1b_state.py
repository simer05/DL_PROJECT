"""Pull cluster state in one SSH session:
- teacher-seed bootstrap completion (count meta.json under exp/cf_fisd/_teachers/tabred_seed{1..4}/)
- headline runs: count of report.json under exp/cf_fisd/tabred/<ds>/<variant>-evaluation/<seed>/
- consensus runs: same
- list any in-progress/queued PBS jobs
"""
from __future__ import annotations
import os, json
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'
DATASETS = ['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
HEADLINE = ['baseline_plr','hetero_raw_lam0.05','hetero_raw_lam0.1','hetero_raw_lam0.2']
CONSENSUS = ['consensus_raw_lam0.05','consensus_raw_lam0.1','consensus_raw_lam0.2']
ABL = ['hetero_softmax_lam0.05','hetero_softmax_lam0.1','hetero_softmax_lam0.2',
       'hetero_l1norm_lam0.05','hetero_l1norm_lam0.1','hetero_l1norm_lam0.2'] + \
      [f'homo_{t}_raw_lam{l}' for t in ('xgb','lgbm','cat') for l in (0.05,0.1,0.2)]


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def run(tgt, cmd, timeout=120):
    _, so, se = tgt.exec_command(cmd, timeout=timeout)
    return so.read().decode('utf-8','replace'), se.read().decode('utf-8','replace'), so.channel.recv_exit_status()


def main():
    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    gw, tgt = connect(pw)
    try:
        # 1) Teacher-seed bootstrap: meta.json under _teachers/tabred_seed{1..4}/<ds>/<variant>/meta.json
        # Variants are xgb_default / lgbm_default / cat_default (per cf_fisd_teachers.py)
        # Each (seed, dataset) job produces all 3 teachers.
        cmd = (f'for s in 1 2 3 4; do for d in {" ".join(DATASETS)}; do '
               f'  for t in xgb_default lgbm_default cat_default; do '
               f'    p="{PD}/exp/cf_fisd/_teachers/tabred_seed${{s}}/${{d}}/${{t}}/meta.json"; '
               f'    if [ -e "$p" ]; then echo "TS $s $d $t DONE"; else echo "TS $s $d $t MISS"; fi; '
               f'  done; '
               f'done; done')
        ts_out, _, _ = run(tgt, cmd, timeout=120)

        # 2) Headline 15-seed: report.json under tabred/<ds>/<variant>-evaluation/<seed>/report.json
        cmd2_lines = []
        for ds in DATASETS:
            for v in HEADLINE:
                # 15 seeds for headline
                cmd2_lines.append(f'for s in $(seq 0 14); do p="{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/$s/report.json"; '
                                  f'if [ -e "$p" ]; then echo "H {ds} {v} $s DONE"; else echo "H {ds} {v} $s MISS"; fi; done')
        head_out, _, _ = run(tgt, '\n'.join(cmd2_lines), timeout=300)

        # 3) Consensus runs (which seeds are present per cell)
        cmd3_lines = []
        for ds in DATASETS:
            for v in CONSENSUS:
                cmd3_lines.append(f'for s in $(seq 0 14); do p="{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/$s/report.json"; '
                                  f'if [ -e "$p" ]; then echo "C {ds} {v} $s DONE"; else echo "C {ds} {v} $s MISS"; fi; done')
        cons_out, _, _ = run(tgt, '\n'.join(cmd3_lines), timeout=300)

        # 4) Ablation variants seed counts (just count completed)
        cmd4_lines = []
        for ds in DATASETS:
            for v in ABL:
                cmd4_lines.append(f'for s in $(seq 0 4); do p="{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/$s/report.json"; '
                                  f'if [ -e "$p" ]; then echo "A {ds} {v} $s DONE"; else echo "A {ds} {v} $s MISS"; fi; done')
        abl_out, _, _ = run(tgt, '\n'.join(cmd4_lines), timeout=300)

        # 5) qstat
        qs, _, _ = run(tgt, 'qstat -u simerjit')

        # Aggregate
        report = {
            'teacher_seed_lines': ts_out.splitlines(),
            'headline_lines': head_out.splitlines(),
            'consensus_lines': cons_out.splitlines(),
            'ablation_lines': abl_out.splitlines(),
            'qstat': qs,
        }
        out_path = r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated\step1b_state.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        # Quick summary
        ts_done = sum(1 for l in ts_out.splitlines() if l.endswith('DONE'))
        ts_total = len(ts_out.splitlines())
        head_done = sum(1 for l in head_out.splitlines() if l.endswith('DONE'))
        head_total = len(head_out.splitlines())
        cons_done = sum(1 for l in cons_out.splitlines() if l.endswith('DONE'))
        cons_total = len(cons_out.splitlines())
        abl_done = sum(1 for l in abl_out.splitlines() if l.endswith('DONE'))
        abl_total = len(abl_out.splitlines())

        print(f'TEACHER-SEED bootstrap: {ts_done}/{ts_total} teachers done ({ts_done/ts_total*100:.0f}%)')
        # Group teacher-seed by (seed, dataset) — each job produces 3 teachers, so check all 3
        ts_jobs_done = {}
        for l in ts_out.splitlines():
            parts = l.split()
            if len(parts) >= 5 and parts[0] == 'TS':
                key = (parts[1], parts[2])  # (seed, dataset)
                ts_jobs_done.setdefault(key, []).append(parts[4])  # DONE/MISS
        n_jobs_complete = sum(1 for k, v in ts_jobs_done.items() if all(x=='DONE' for x in v))
        n_jobs_total = len(ts_jobs_done)
        print(f'TEACHER-SEED jobs (all 3 teachers done): {n_jobs_complete}/{n_jobs_total}')

        print(f'HEADLINE 15-seed: {head_done}/{head_total} reports')
        # Per-variant
        head_per = {}
        for l in head_out.splitlines():
            parts = l.split()
            if len(parts) >= 5 and parts[0] == 'H':
                k = (parts[1], parts[2])
                head_per[k] = head_per.get(k, 0) + (1 if parts[4]=='DONE' else 0)
        for (ds, v), n in sorted(head_per.items()):
            print(f'  {ds:25s} {v:25s} {n}/15')

        print(f'CONSENSUS: {cons_done}/{cons_total} reports')
        cons_per = {}
        for l in cons_out.splitlines():
            parts = l.split()
            if len(parts) >= 5 and parts[0] == 'C':
                k = (parts[1], parts[2])
                cons_per[k] = cons_per.get(k, 0) + (1 if parts[4]=='DONE' else 0)
        for (ds, v), n in sorted(cons_per.items()):
            print(f'  {ds:25s} {v:25s} {n}/15')

        print(f'ABLATIONS: {abl_done}/{abl_total} reports')
        abl_per = {}
        for l in abl_out.splitlines():
            parts = l.split()
            if len(parts) >= 5 and parts[0] == 'A':
                k = (parts[1], parts[2])
                abl_per[k] = abl_per.get(k, 0) + (1 if parts[4]=='DONE' else 0)
        for (ds, v), n in sorted(abl_per.items()):
            if n > 0:
                print(f'  {ds:25s} {v:35s} {n}/5')

        print(f'\n=== qstat ===\n{qs}')
        print(f'\nSaved {out_path}')

    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
