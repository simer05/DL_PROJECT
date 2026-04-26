"""Find teacher .npy files."""
from __future__ import annotations
import os
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'


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
        # 1) ls _teachers
        for path in [f'{PD}/exp/cf_fisd/_teachers',
                     f'{PD}/exp/cf_fisd/_teachers/tabred',
                     f'{PD}/exp/cf_fisd/_teachers/tabred/sberbank-housing',
                     f'{PD}/exp/cf_fisd/_teachers/tabred_consensus',
                     f'{PD}/exp/cf_fisd/_teachers/tabred_consensus/sberbank-housing']:
            o, _, _ = run(tgt, f'ls -la {path} 2>&1 | head -30')
            print(f'\n=== {path} ===')
            print(o)
    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
