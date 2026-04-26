"""Get full Output_Path and Variable_List for one failed job."""
from __future__ import annotations
import os
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'


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
        # awk -based extraction unwraps PBS folded fields properly
        for jid in ('13910545','13910549','13910568'):
            print(f'\n=== {jid} ===')
            o, _, _ = run(tgt, f"qstat -fx {jid} | awk '/^[A-Z]/{{print \"\"; printf \"%s\", $0; next}} /^\\t/{{sub(/^\\t/,\"\"); printf \"%s\", $0}} END{{print \"\"}}' | grep -E 'Output_Path|Error_Path|Variable_List|exit_status|comment'")
            print(o)
        # also dump the full env for the running jobs to see what (DS, SEED) they cover
        for jid in ('13910568','13910569'):
            print(f'\n=== running {jid} env ===')
            o, _, _ = run(tgt, f"qstat -f {jid} | awk '/^[A-Z]/{{print \"\"; printf \"%s\", $0; next}} /^\\t/{{sub(/^\\t/,\"\"); printf \"%s\", $0}} END{{print \"\"}}' | grep -E 'Variable_List|Output_Path'")
            print(o[:2000])
        # Also list the throttler logs
        o, _, _ = run(tgt, 'ls -lt /home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/_throttler_logs/ 2>/dev/null | head -10')
        print('\n=== throttler logs (top 10 newest) ===\n', o)
        o, _, _ = run(tgt, 'find /home/users/ntu/simerjit -name "cf_fisd_tseed.o*" 2>/dev/null | head -5')
        print('\n=== find tseed output files ===\n', o)
        o, _, _ = run(tgt, 'find /home/users/ntu/simerjit -name "cf_fisd_tseed.e*" 2>/dev/null | head -5')
        print('\n=== find tseed error files ===\n', o)
    finally:
        tgt.close(); gw.close()


if __name__ == '__main__':
    main()
