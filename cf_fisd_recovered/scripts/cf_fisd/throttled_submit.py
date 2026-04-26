from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import paramiko

GW_HOST, GW_USER = '172.21.26.100', 'simerjit001'
TGT_HOST, TGT_USER = 'aspire2antu.nscc.sg', 'simerjit'
DEFAULT_PBS_RUN = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_run_one.pbs'
DEFAULT_PBS_TEACHER = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_train_teacher.pbs'
PAPER_DIR_REMOTE = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'
MAX_CONCURRENT = 5
POLL_SEC = 30


def connect(password: str):
    gw = paramiko.SSHClient()
    gw.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gw.connect(GW_HOST, username=GW_USER, password=password, timeout=30,
               allow_agent=False, look_for_keys=False)
    chan = gw.get_transport().open_channel('direct-tcpip', (TGT_HOST, 22), ('127.0.0.1', 0))
    tgt = paramiko.SSHClient()
    tgt.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    tgt.connect(TGT_HOST, username=TGT_USER, password=password, sock=chan, timeout=30,
                allow_agent=False, look_for_keys=False)
    return gw, tgt


def run_cmd(holder, password, cmd, timeout=60):
    last_err = None
    for _ in range(8):
        try:
            _, o, e = holder['tgt'].exec_command(cmd, timeout=timeout)
            return o.read().decode().strip(), e.read().decode().strip(), o.channel.recv_exit_status()
        except (paramiko.SSHException, OSError, EOFError) as ex:
            last_err = ex
            for w in (10, 30, 60, 120, 180):
                try:
                    try:
                        holder['tgt'].close()
                        holder['gw'].close()
                    except Exception:
                        pass
                    gw, tgt = connect(password)
                    holder['gw'] = gw
                    holder['tgt'] = tgt
                    print('  reconnected', flush=True)
                    break
                except Exception:
                    time.sleep(w)
    raise RuntimeError(f'ran out of reconnects: {last_err}')


def in_queue_count(holder, password):
    out, _, _ = run_cmd(holder, password,
                       f'qstat -u {TGT_USER} 2>/dev/null | awk "NR>5" | wc -l')
    try:
        return int(out)
    except Exception:
        return 0


def already_done(holder, password, output_dir_abs: str) -> bool:
    out, _, _ = run_cmd(holder, password,
                       f'test -f "{output_dir_abs}/DONE" && echo Y || echo N')
    return out.strip().endswith('Y')


def build_student_jobs(paper_dir_remote: str, datasets, variants, n_seeds: int):
    jobs = []
    for ds in datasets:
        for v in variants:
            for s in range(n_seeds):
                cfg = f'{paper_dir_remote}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/{s}.toml'
                out_dir = cfg[:-len('.toml')]
                jobs.append({
                    'kind': 'student',
                    'CONFIG_TOML': cfg,
                    'output_dir_abs': out_dir,
                    'name': f'{ds}/{v}/seed{s}',
                })
    return jobs


def build_teacher_jobs(datasets):
    jobs = []
    for ds in datasets:
        out = (
            f'/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper/exp/cf_fisd/'
            f'_teachers/tabred/{ds}/meta.json'
        )
        jobs.append({
            'kind': 'teacher',
            'DATASET': ds,
            'output_dir_abs': out,
            'name': f'teacher/{ds}',
        })
    return jobs


def submit(holder, password, pbs_script: str, env_vars: dict[str, str]):
    vstr = ','.join(f'{k}={v}' for k, v in env_vars.items())
    out, err, rc = run_cmd(holder, password, f"qsub -v '{vstr}' {pbs_script}")
    return out, err, rc


def main_inner(args):
    password = os.environ.get('NSCC_PW') or os.environ.get('NSCC_PASSWORD')
    if not password:
        print('ERROR: set NSCC_PW (or NSCC_PASSWORD) env var', file=sys.stderr)
        sys.exit(2)

    if args.kind == 'student':
        jobs = build_student_jobs(
            args.paper_dir_remote, args.datasets, args.variants, args.n_seeds
        )
        pbs = args.pbs_script or DEFAULT_PBS_RUN
    else:
        jobs = build_teacher_jobs(args.datasets)
        pbs = args.pbs_script or DEFAULT_PBS_TEACHER

    print(f'connecting to {TGT_HOST}...', flush=True)
    gw, tgt = connect(password)
    holder = {'gw': gw, 'tgt': tgt}

    pending = []
    skipped_done = 0
    for j in jobs:
        if not args.force and already_done(holder, password, j['output_dir_abs']):
            skipped_done += 1
            continue
        pending.append(j)

    print(
        f'kind={args.kind}  total={len(jobs)}  done={skipped_done}  pending={len(pending)}',
        flush=True,
    )
    if not pending:
        return

    t0 = time.time()
    submitted = 0
    while pending:
        in_q = in_queue_count(holder, password)
        slots = max(0, MAX_CONCURRENT - in_q)
        if slots > 0:
            for _ in range(min(slots, len(pending))):
                j = pending.pop(0)
                env = {k: v for k, v in j.items() if k.isupper()}
                out, err, rc = submit(holder, password, pbs, env)
                if rc != 0:
                    print(f'  qsub FAILED for {j["name"]}: {out} {err}', flush=True)
                    pending.append(j)
                    break
                submitted += 1
                jid = out.split('.')[0] if out else '?'
                print(
                    f'  [{submitted:3d}] {j["name"]:<60} -> {jid}  '
                    f'(in_q={in_q+1} pending={len(pending)} elapsed={int(time.time()-t0)}s)',
                    flush=True,
                )
                in_q += 1
        time.sleep(POLL_SEC)
    while in_queue_count(holder, password) > 0:
        print(f'  draining: {in_queue_count(holder, password)} in queue', flush=True)
        time.sleep(POLL_SEC)
    print(f'done. elapsed={int(time.time()-t0)}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['student', 'teacher'], required=True)
    ap.add_argument(
        '--datasets',
        nargs='+',
        default=[
            'sberbank-housing', 'ecom-offers', 'homesite-insurance',
            'cooking-time', 'delivery-eta',
        ],
    )
    ap.add_argument(
        '--variants',
        nargs='+',
        default=['baseline_plr', 'hetero_raw_lam0.05', 'hetero_raw_lam0.1', 'hetero_raw_lam0.2'],
    )
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--paper-dir-remote', default=PAPER_DIR_REMOTE)
    ap.add_argument('--pbs-script', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    main_inner(args)


if __name__ == '__main__':
    main()
