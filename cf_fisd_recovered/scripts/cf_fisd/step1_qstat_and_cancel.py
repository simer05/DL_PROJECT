"""Step 1: qstat -> identify -> qdel non-essential jobs.

Keeps only:
- Teacher-seed bootstrap jobs 13910545-13910569
- Consensus 15-seed extension IF symmetric (5 datasets x 3 lambdas)

Cancels:
- softmax / l1norm / homo variants
- consensus 15-seed extension if cherry-picked
"""
from __future__ import annotations
import os, re, sys, json
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

KEEP_JOBS = set(str(j) for j in range(13910545, 13910570))  # teacher-seed bootstrap


def connect(pw):
    gw=paramiko.SSHClient(); gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=pw, allow_agent=False, look_for_keys=False, timeout=30)
    chan=gw.get_transport().open_channel('direct-tcpip',(TGT_HOST,22),('127.0.0.1',0))
    tgt=paramiko.SSHClient(); tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=pw, sock=chan, allow_agent=False, look_for_keys=False, timeout=30)
    return gw, tgt


def run(tgt, cmd, timeout=60):
    _, so, se = tgt.exec_command(cmd, timeout=timeout)
    return so.read().decode('utf-8','replace'), se.read().decode('utf-8','replace'), so.channel.recv_exit_status()


def main():
    pw = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not pw: raise SystemExit('Set NSCC_PW')
    gw, tgt = connect(pw)
    try:
        # 1) qstat full to get job names with -f
        out, err, rc = run(tgt, 'qstat -u simerjit -t')
        print('=== qstat -u simerjit -t ===')
        print(out)
        print(err)

        # parse: lines look like:
        # 13910545.pbs101    simerjit normal   ts_xxx        12345    1   4   64gb 02:00 R 00:30
        jobs = []
        for line in out.splitlines():
            m = re.match(r'^(\d+)\.\S+\s+(\S+)\s+(\S+)\s+(\S+)\s', line)
            if m and m.group(2) == 'simerjit':
                jobs.append({'jobid': m.group(1), 'queue': m.group(3), 'name': m.group(4), 'raw': line})
        print(f'Parsed {len(jobs)} jobs.')

        # 2) Get full job names via qstat -f if name appears truncated
        # Also categorize each
        keep = []
        kill = []
        consensus_jobs = []
        for j in jobs:
            jid = j['jobid']; name = j['name']
            if jid in KEEP_JOBS:
                keep.append(j); continue
            # Pull -f to get the full name + variant
            fout, _, _ = run(tgt, f'qstat -f {jid}')
            jname = ''
            for l in fout.splitlines():
                if 'Job_Name' in l and '=' in l:
                    jname = l.split('=',1)[1].strip(); break
            j['fullname'] = jname
            ln = jname.lower()
            if 'softmax' in ln or 'l1norm' in ln or 'homo_' in ln:
                kill.append(j)
            elif 'consensus' in ln:
                consensus_jobs.append(j)
            elif 'teacher' in ln or 'tseed' in ln or 'ts_' in ln:
                keep.append(j)
            elif 'baseline_plr' in ln or 'hetero_raw' in ln:
                keep.append(j)  # the headline runs
            else:
                # unknown — keep for safety, log
                j['unknown'] = True
                keep.append(j)

        # 3) Decide consensus symmetry — count by (dataset, lambda)
        # Job naming convention may be: cf_<dataset>_<variant>_s<seed>
        cells = {}
        for j in consensus_jobs:
            n = j['fullname'].lower()
            ds_match = re.search(r'(sberbank-housing|ecom-offers|homesite-insurance|cooking-time|delivery-eta)', n)
            lam_match = re.search(r'lam(0\.\d+)', n)
            ds = ds_match.group(1) if ds_match else 'unknown'
            lam = lam_match.group(1) if lam_match else 'unknown'
            cells.setdefault((ds, lam), []).append(j['jobid'])

        DATASETS = ['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
        LAMS = ['0.05','0.1','0.2']
        expected = {(d,l) for d in DATASETS for l in LAMS}
        present = set(cells.keys())
        missing = expected - present
        # Symmetric means: every (ds, lam) cell has at least 1 job  OR  finished cells exist completed
        # But we should also check the report.json count for already-completed seeds.
        # For now: if all 15 cells appear in queue OR no consensus jobs in queue (already done), treat as symmetric.

        if not consensus_jobs:
            # No consensus jobs queued — they may have already completed. Need to check filesystem.
            consensus_decision = 'no-consensus-in-queue'
        elif len(present) == 15:
            consensus_decision = 'symmetric-keep'
        else:
            consensus_decision = f'asymmetric-cancel: missing {sorted(missing)}'
            kill.extend(consensus_jobs)
            consensus_jobs = []

        # If symmetric, keep them
        if consensus_decision == 'symmetric-keep':
            keep.extend(consensus_jobs)

        print('=== DECISION ===')
        print(f'Keep: {len(keep)} jobs')
        print(f'Kill: {len(kill)} jobs')
        print(f'Consensus decision: {consensus_decision}')

        # 4) qdel
        if kill:
            kill_ids = [j['jobid'] for j in kill]
            # qdel many at once is fine
            for batch_start in range(0, len(kill_ids), 50):
                batch = kill_ids[batch_start:batch_start+50]
                cmd = 'qdel ' + ' '.join(batch)
                qout, qerr, qrc = run(tgt, cmd, timeout=120)
                print(f'qdel batch {batch_start}: rc={qrc}')
                if qout: print(qout)
                if qerr: print(qerr)

        # 5) Re-qstat to confirm
        out2, _, _ = run(tgt, 'qstat -u simerjit')
        print('=== qstat AFTER qdel ===')
        print(out2)

        # Save report
        report = {
            'kept': [{'jobid': j['jobid'], 'name': j.get('fullname', j['name'])} for j in keep],
            'killed': [{'jobid': j['jobid'], 'name': j.get('fullname', j['name'])} for j in kill],
            'consensus_decision': consensus_decision,
            'consensus_cells': {f'{k[0]}/{k[1]}': v for k,v in cells.items()},
            'qstat_after': out2,
        }
        out_path = r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated\step1_cancel_report.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'Saved {out_path}')
        print(f'\nSUMMARY: cancelled {len(kill)} jobs; remaining {len(keep)} jobs (incl. {len([j for j in keep if j["jobid"] in KEEP_JOBS])} teacher-seed)')

    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
