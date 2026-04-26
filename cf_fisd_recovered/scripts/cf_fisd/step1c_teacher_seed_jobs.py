"""Identify which (dataset, seed) each teacher-seed job covers + check failed history."""
from __future__ import annotations
import os, json, re
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

RUNNING = ['13910555','13910557','13910558','13910559','13910563','13910566','13910567','13910568','13910569']
HISTORICAL = ['13910545','13910546','13910547','13910548','13910549','13910550','13910551','13910552','13910553','13910554','13910556','13910560','13910561','13910562','13910564','13910565']


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
        out_lines = []
        # 1) Get full job names + status for running
        for jid in RUNNING:
            o, _, _ = run(tgt, f'qstat -f {jid} 2>&1 | grep -E "^[[:space:]]*(Job_Name|job_state|exit_status|resources_used\\.walltime|Output_Path)"')
            out_lines.append(f'\n=== {jid} (RUNNING) ===\n{o}')

        # 2) Get historical job statuses (use -fx for finished)
        for jid in HISTORICAL[:10]:  # check first 10 historical
            o, _, _ = run(tgt, f'qstat -fx {jid} 2>&1 | grep -E "^[[:space:]]*(Job_Name|job_state|exit_status|resources_used\\.walltime|comment)" | head -20')
            out_lines.append(f'\n=== {jid} (historical) ===\n{o}')

        report = '\n'.join(out_lines)
        out_path = r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated\step1c_jobs_detail.txt'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f: f.write(report)
        print(report)
        print(f'\nSaved {out_path}')

    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
