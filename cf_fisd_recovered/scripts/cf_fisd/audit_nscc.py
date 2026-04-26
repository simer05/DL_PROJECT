"""Comprehensive NSCC + bug audit:
1. Verify nscc_run_one.pbs uses personal-simerjit + normal queue
2. Verify nscc_train_teacher_seed.pbs uses personal-simerjit + normal queue
3. Verify the 5 FP32 toml configs we wrote are valid (parse + amp=false + seed correct)
4. Sample-verify headline numbers by re-pulling 3 specific reports
5. Verify TabM training code respects amp=false (check the code path)
6. Check current qstat for any unexpected jobs
"""
from __future__ import annotations
import os, json
import paramiko
import tomllib

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
    sftp = tgt.open_sftp()
    audit = {}
    try:
        # === 1. nscc_run_one.pbs ===
        with sftp.open(f'/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_run_one.pbs') as f:
            txt = f.read().decode('utf-8')
        print('=== nscc_run_one.pbs ===')
        print(txt[:1500])
        audit['nscc_run_one'] = {
            'has_personal_simerjit': '#PBS -P personal-simerjit' in txt,
            'has_normal_queue': '#PBS -q normal' in txt,
            'uses_43001002': '43001002' in txt,
            'uses_gdev_q': '-q gdev' in txt,
            'uses_ai_q': '-q ai' in txt,
        }
        print('  ✓ -P personal-simerjit:', audit['nscc_run_one']['has_personal_simerjit'])
        print('  ✓ -q normal:', audit['nscc_run_one']['has_normal_queue'])
        print('  ✗ uses 43001002:', audit['nscc_run_one']['uses_43001002'])

        # === 2. nscc_train_teacher_seed.pbs ===
        with sftp.open(f'/home/users/ntu/simerjit/scratch/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_train_teacher_seed.pbs') as f:
            txt = f.read().decode('utf-8')
        print('\n=== nscc_train_teacher_seed.pbs ===')
        print(txt[:1500])
        audit['nscc_train_teacher_seed'] = {
            'has_personal_simerjit': '#PBS -P personal-simerjit' in txt,
            'has_normal_queue': '#PBS -q normal' in txt,
            'uses_43001002': '43001002' in txt,
        }
        print('  ✓ -P personal-simerjit:', audit['nscc_train_teacher_seed']['has_personal_simerjit'])
        print('  ✓ -q normal:', audit['nscc_train_teacher_seed']['has_normal_queue'])

        # === 3. Validate FP32 tomls ===
        print('\n=== FP32 tomls ===')
        audit['fp32_tomls'] = []
        for s in range(5):
            p = f'{PD}/exp/cf_fisd/tabred/cooking-time/baseline_plr_fp32-evaluation/{s}.toml'
            with sftp.open(p) as f:
                content = f.read().decode('utf-8')
            try:
                parsed = tomllib.loads(content)
                ok = (parsed.get('seed') == s and parsed.get('amp') is False)
                print(f'  seed{s}: parsed OK; seed={parsed.get("seed")}, amp={parsed.get("amp")}  {"✓" if ok else "✗"}')
                audit['fp32_tomls'].append({'seed': s, 'parsed_seed': parsed.get('seed'), 'amp': parsed.get('amp'), 'ok': ok})
            except Exception as e:
                print(f'  seed{s}: PARSE ERROR {e}')
                audit['fp32_tomls'].append({'seed': s, 'error': str(e)})

        # === 4. Sample-verify 3 headline reports ===
        print('\n=== Sample report verification ===')
        samples = [
            ('homesite-insurance', 'hetero_raw_lam0.05', 0),
            ('homesite-insurance', 'hetero_raw_lam0.2', 7),
            ('cooking-time', 'baseline_plr', 5),
        ]
        audit['report_samples'] = []
        for ds, v, s in samples:
            p = f'{PD}/exp/cf_fisd/tabred/{ds}/{v}-evaluation/{s}/report.json'
            with sftp.open(p) as f:
                r = json.load(f)
            test_score = r['metrics']['test']['score']
            val_score = r['metrics']['val']['score']
            best_step = r.get('best_step')
            time_s = r.get('time')
            amp = r.get('amp_dtype')
            print(f'  {ds}/{v}/seed{s}: test={test_score:+.6f}, val={val_score:+.6f}, best_step={best_step}, time={time_s}, amp={amp}')
            audit['report_samples'].append({'ds': ds, 'v': v, 's': s,
                                            'test': test_score, 'val': val_score,
                                            'best_step': best_step, 'amp_dtype': amp})

        # === 5. Inspect TabM training code path for amp ===
        print('\n=== amp handling in training code ===')
        # The training script is paper/bin/model.py per CLAUDE.md
        try:
            with sftp.open(f'{PD}/bin/model.py') as f:
                code = f.read().decode('utf-8')
            # Find amp references
            import re
            amp_lines = [(i+1, l) for i, l in enumerate(code.splitlines())
                         if re.search(r'\b(amp|autocast|bfloat16|float32|float16)\b', l)]
            print(f'  found {len(amp_lines)} amp-related lines in bin/model.py:')
            for ln, l in amp_lines[:30]:
                print(f'    L{ln}: {l.strip()[:140]}')
            audit['amp_lines_count'] = len(amp_lines)
            audit['amp_lines_sample'] = [{'line': ln, 'text': l.strip()} for ln, l in amp_lines[:30]]
        except FileNotFoundError:
            print('  bin/model.py not found at expected path; skipping')

        # === 6. Current qstat ===
        out, _, _ = run(tgt, 'qstat -u simerjit')
        print('\n=== qstat NOW ===')
        print(out)
        audit['qstat_now'] = out

        # === 7. Allocation usage ===
        out, err, _ = run(tgt, 'qsu personal-simerjit 2>&1 || echo "qsu not available"')
        print('\n=== personal-simerjit allocation ===')
        print(out)
        out, err, _ = run(tgt, 'pbsstat -u simerjit 2>&1 | head -10 || echo "pbsstat not available"')
        print(out)
        out, err, _ = run(tgt, 'mybalance 2>&1 | head -10 || echo "mybalance not available"')
        print(out)

        out_path = r'C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated\AUDIT.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(audit, f, indent=2, default=str)
        print(f'\nWrote {out_path}')
    finally:
        sftp.close(); tgt.close(); gw.close()


if __name__ == '__main__':
    main()
