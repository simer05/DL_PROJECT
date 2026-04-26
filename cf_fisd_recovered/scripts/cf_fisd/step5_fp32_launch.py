"""Step 5: launch cooking-time baseline_plr with AMP=fp32, 5 seeds.

Strategy:
1. Read the existing cooking-time baseline_plr config TOML
2. Create a new variant config with amp_dtype=float32 (or amp=disabled)
3. Generate 5 seed configs
4. qsub via the existing nscc_run_one.pbs script
5. Verify submission count
"""
from __future__ import annotations
import os, re, time
import paramiko

GW_HOST='172.21.26.100'; GW_USER='simerjit001'
TGT_HOST='aspire2antu.nscc.sg'; TGT_USER='simerjit'
PD = '/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/paper'
N_SEEDS = 5


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
        # 1) read cooking-time baseline_plr seed-0 config
        src_cfg = f'{PD}/exp/cf_fisd/tabred/cooking-time/baseline_plr-evaluation/0.toml'
        with sftp.open(src_cfg) as f:
            cfg_text = f.read().decode('utf-8')
        print('=== source config ===')
        print(cfg_text[:2000])

        # 2) create a new variant directory
        new_var = 'baseline_plr_fp32'
        new_dir = f'{PD}/exp/cf_fisd/tabred/cooking-time/{new_var}-evaluation'
        run(tgt, f'mkdir -p {new_dir}')

        # 3) generate 5 seed configs with amp disabled (forces FP32)
        for s in range(N_SEEDS):
            new_text = cfg_text
            # Disable AMP — top-level `amp = true` -> `amp = false`
            new_text = re.sub(r'^amp\s*=\s*true', 'amp = false', new_text, flags=re.M | re.I)
            # also blanket-disable amp_dtype if present
            new_text = re.sub(r'amp_dtype\s*=\s*[\'"]?[^\'"\n]*[\'"]?',
                              "amp_dtype = 'float32'", new_text)
            # change seed
            if re.search(r'^seed\s*=\s*\d+', new_text, flags=re.M):
                new_text = re.sub(r'^seed\s*=\s*\d+', f'seed = {s}', new_text, flags=re.M)
            else:
                new_text = f'seed = {s}\n' + new_text
            new_cfg_path = f'{new_dir}/{s}.toml'
            with sftp.open(new_cfg_path, 'w') as f:
                f.write(new_text)
            print(f'wrote {new_cfg_path}')

        # 4) Print one final config to verify
        with sftp.open(f'{new_dir}/0.toml') as f:
            print('\n=== final 0.toml (first 1500 chars) ===')
            print(f.read().decode('utf-8')[:1500])

        # 5) submit via nscc_run_one.pbs (existing PBS script)
        # qsub -v 'CONFIG_TOML=...' nscc_run_one.pbs
        for s in range(N_SEEDS):
            cfg = f'{new_dir}/{s}.toml'
            cmd = f"qsub -v 'CONFIG_TOML={cfg}' /home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_run_one.pbs"
            out, err, rc = run(tgt, cmd, timeout=60)
            print(f'seed {s}: rc={rc}  out={out.strip()}  err={err.strip()}')

        # 6) verify
        time.sleep(2)
        out, _, _ = run(tgt, 'qstat -u simerjit | tail -25')
        print('\n=== qstat after submission ===')
        print(out)

    finally:
        sftp.close(); tgt.close(); gw.close()


if __name__ == '__main__':
    main()
