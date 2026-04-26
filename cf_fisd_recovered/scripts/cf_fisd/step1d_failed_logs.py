"""Check .o output of failed teacher-seed jobs to know if relaunch is needed."""
from __future__ import annotations
import os
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

FAILED = ['13910545','13910546','13910547','13910548','13910549','13910550','13910551','13910552','13910553','13910554','13910556','13910560','13910561','13910562','13910564','13910565']


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
        # Look for output files in user home dir or scratch
        # Default PBS output goes to ~/cf_fisd_tseed.o<jobid>
        # Check both ~ and scratch
        for jid in FAILED[:5]:  # sample first 5
            print(f'\n=== {jid} ===')
            o, _, _ = run(tgt, f'ls -la ~/cf_fisd_tseed.o{jid} ~/cf_fisd_tseed.e{jid} '
                               f'/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/cf_fisd_tseed.o{jid} '
                               f'/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/cf_fisd_tseed.e{jid} 2>&1 | head -10')
            print(o)
            # Try to find the output file via qstat
            qf, _, _ = run(tgt, f'qstat -fx {jid} 2>&1 | grep -E "Output_Path|Error_Path|exit_status|Variable_List" | head -5')
            print('qstat -fx:', qf)
        print('\n=== exit_status of all failed ===')
        exits = []
        for jid in FAILED:
            ex, _, _ = run(tgt, f'qstat -fx {jid} 2>&1 | grep -E "exit_status|Variable_List" | tr -d "\\n\\t " ')
            exits.append(f'{jid}: {ex[:200]}')
        for e in exits:
            print(e)
    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
