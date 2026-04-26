"""Tail of failed teacher-seed jobs."""
from __future__ import annotations
import os
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
LOGS = '/home/users/ntu/simerjit/scratch/tabm_proj/logs/cf_fisd'


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
        for jid in ('13910545','13910549','13910551','13910560','13910562'):
            print(f'\n=== HEAD {jid} ===')
            o, _, _ = run(tgt, f'head -30 {LOGS}/{jid}.pbs101.OU')
            print(o)
            print(f'\n=== TAIL {jid} ===')
            o, _, _ = run(tgt, f'tail -40 {LOGS}/{jid}.pbs101.OU')
            print(o)
    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
