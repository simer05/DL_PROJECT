"""Read PBS script + throttler + check for output files using simple commands."""
from __future__ import annotations
import os
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'

PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork'


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
        # 1) PBS script
        try:
            with sftp.open(f'{PD}/scripts/cf_fisd/nscc_train_teacher_seed.pbs') as f:
                print('=== PBS script ===')
                print(f.read().decode('utf-8'))
        except FileNotFoundError:
            print('PBS script not at expected path; trying alternates...')
            for p in [f'{PD}/scripts/cf_fisd/nscc_teacher_seed.pbs',
                      f'{PD}/scripts/cf_fisd/teacher_seed.pbs']:
                try:
                    with sftp.open(p) as f:
                        print(f'Found at {p}')
                        print(f.read().decode('utf-8')); break
                except FileNotFoundError: pass

        # 2) Throttler
        try:
            with sftp.open(f'{PD}/scripts/cf_fisd/throttled_submit.py') as f:
                print('\n=== throttled_submit.py (first 100 lines) ===')
                print('\n'.join(f.read().decode('utf-8').splitlines()[:120]))
        except FileNotFoundError as e:
            print(f'throttled_submit.py not found: {e}')

        # 3) Find output files (use ls instead of find — faster)
        for p in [
            '/home/users/ntu/simerjit/cf_fisd_tseed.o13910545',
            '/home/users/ntu/simerjit/cf_fisd_tseed.e13910545',
            f'{PD}/scripts/cf_fisd/cf_fisd_tseed.o13910545',
            f'{PD}/scripts/cf_fisd/cf_fisd_tseed.e13910545',
            f'{PD}/cf_fisd_tseed.o13910545',
        ]:
            try:
                st = sftp.stat(p)
                print(f'EXISTS: {p} size={st.st_size}')
                if st.st_size < 50000:
                    with sftp.open(p) as f:
                        print(f.read().decode('utf-8','replace')[:5000])
            except FileNotFoundError: pass

        # 4) ls scripts dir for output files
        try:
            for entry in sftp.listdir(f'{PD}/scripts/cf_fisd'):
                if 'tseed.o' in entry or 'tseed.e' in entry:
                    print(f'  {entry}')
        except: pass
        try:
            for entry in sftp.listdir('/home/users/ntu/simerjit'):
                if 'tseed' in entry:
                    print(f'  HOME: {entry}')
        except: pass

    finally:
        sftp.close(); tgt.close(); gw.close()


if __name__ == '__main__':
    main()
