"""Read PBS output logs from /home/users/ntu/simerjit/scratch/tabm_proj/logs/cf_fisd/."""
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
    sftp = tgt.open_sftp()
    try:
        # list logs newest first
        out, _, _ = run(tgt, f'ls -lt {LOGS}/ 2>&1 | head -40')
        print('=== logs dir ===')
        print(out)

        # Read tail of one failed job (use the actual file name pattern)
        for jid in ('13910545','13910546','13910549','13910568'):
            out, _, _ = run(tgt, f'ls {LOGS}/cf_fisd_tseed.o{jid} 2>/dev/null')
            if out.strip():
                print(f'\n=== log for {jid} ===')
                tail, _, _ = run(tgt, f'cat {LOGS}/cf_fisd_tseed.o{jid}')
                print(tail[:6000])
            else:
                # maybe in different naming
                out2, _, _ = run(tgt, f'ls {LOGS}/ 2>&1 | grep {jid}')
                print(f'\n=== {jid}: not found, alts: {out2.strip()} ===')

    finally:
        sftp.close(); tgt.close(); gw.close()


if __name__ == '__main__':
    main()
