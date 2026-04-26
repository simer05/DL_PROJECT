from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DATASETS = (
    'sberbank-housing',
    'ecom-offers',
    'homesite-insurance',
    'cooking-time',
    'delivery-eta',
)
PINNED_COMMIT_PREFIX = '28e47ae'
CANONICAL_TRAIN_VAL_TEST_TOTAL = {
    'sberbank-housing': 28321,
    'ecom-offers': 160057,
    'homesite-insurance': 260753,
    'cooking-time': 319986,
    'delivery-eta': 350516,
}


class AuditFinding:
    __slots__ = ('rule', 'ok', 'message')

    def __init__(self, rule: str, ok: bool, message: str):
        self.rule = rule
        self.ok = ok
        self.message = message


def check_pinned_commit(repo_root: Path) -> AuditFinding:
    out = subprocess.run(
        ['git', 'log', '--oneline', '--all'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout
    has_pin = any(line.startswith(PINNED_COMMIT_PREFIX) for line in out.splitlines())
    return AuditFinding(
        'Rule 1', has_pin,
        f'pinned commit {PINNED_COMMIT_PREFIX}* present in git log' if has_pin
        else f'pinned commit {PINNED_COMMIT_PREFIX}* NOT found',
    )


def check_branch(repo_root: Path) -> AuditFinding:
    branch = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    ok = branch == 'cf-fisd-distill'
    return AuditFinding('Rule 2', ok, f'branch={branch!r}')


def check_no_forbidden_edits(repo_root: Path) -> AuditFinding:
    diff = subprocess.run(
        ['git', 'diff', f'{PINNED_COMMIT_PREFIX}', '--name-only'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    forbidden = [
        'paper/lib/data.py',
        'paper/bin/evaluate.py',
        'paper/lib/metrics.py',
    ]
    bad = [f for f in diff if f in forbidden]
    return AuditFinding(
        'Rule 3', not bad,
        'no forbidden edits' if not bad else f'forbidden files modified: {bad}',
    )


def check_module_files_namespaced(repo_root: Path) -> AuditFinding:
    needed = (
        'paper/lib/cf_fisd.py',
        'paper/bin/cf_fisd_teachers.py',
    )
    missing = [n for n in needed if not (repo_root / n).exists()]
    return AuditFinding(
        'Rule 4', not missing,
        'module files in canonical locations' if not missing
        else f'missing namespaced files: {missing}',
    )


def check_state_stamp(repo_root: Path) -> AuditFinding:
    p = repo_root / 'paper' / '.cf_fisd_state.json'
    if not p.exists():
        return AuditFinding('Rule 5', False, 'paper/.cf_fisd_state.json missing')
    try:
        s = json.loads(p.read_text())
    except Exception as e:
        return AuditFinding('Rule 5', False, f'state stamp unreadable: {e}')
    has_keys = all(k in s for k in ('module', 'upstream_commit', 'module_file_shas'))
    return AuditFinding('Rule 5', has_keys,
                        f"stamp present, upstream_commit={s.get('upstream_commit', '?')[:8]}")


def check_five_datasets(repo_root: Path) -> AuditFinding:
    cfg_root = repo_root / 'paper' / 'exp' / 'cf_fisd' / 'tabred'
    if not cfg_root.exists():
        return AuditFinding('Rule 11', False, 'paper/exp/cf_fisd/tabred missing')
    actual = sorted(p.name for p in cfg_root.iterdir() if p.is_dir())
    expected = sorted(DATASETS)
    return AuditFinding(
        'Rule 11', actual == expected,
        f'datasets={actual}' if actual == expected
        else f'expected {expected}, got {actual}',
    )


def check_inheritance_per_dataset(repo_root: Path) -> AuditFinding:
    issues: list[str] = []
    for ds in DATASETS:
        paper = repo_root / 'paper' / 'exp' / 'tabm-piecewiselinear' / 'tabred' / ds / '0-evaluation' / '0.toml'
        ours_baseline = repo_root / 'paper' / 'exp' / 'cf_fisd' / 'tabred' / ds / 'baseline_plr-evaluation' / '0.toml'
        if not paper.exists():
            issues.append(f'{ds}: paper TOML missing')
            continue
        if not ours_baseline.exists():
            issues.append(f'{ds}: baseline_plr TOML missing')
            continue
        ptxt = paper.read_text().strip()
        otxt = ours_baseline.read_text().strip()
        if ptxt != otxt:
            issues.append(f'{ds}: baseline_plr TOML differs from paper TOML')
    return AuditFinding(
        'Rule 14, 19', not issues,
        '5/5 datasets: baseline_plr matches paper template byte-for-byte'
        if not issues else '; '.join(issues),
    )


def check_lambda_zero_off_state(repo_root: Path) -> AuditFinding:
    test_path = repo_root / 'paper' / 'test_cf_fisd.py'
    if not test_path.exists():
        return AuditFinding('Rule 27, 28', False, 'paper/test_cf_fisd.py missing')
    try:
        out = subprocess.run(
            [sys.executable, 'test_cf_fisd.py'],
            cwd=repo_root / 'paper', capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return AuditFinding('Rule 27, 28', False, f'test runner failed: {e}')
    failed = 'FAIL' in out.stdout or out.returncode != 0
    return AuditFinding(
        'Rule 27, 28', not failed,
        'all 4 unit tests pass' if not failed else f'test failures:\n{out.stdout}\n{out.stderr}',
    )


def check_pbs_billing(repo_root: Path) -> AuditFinding:
    pbs_dir = repo_root / 'scripts' / 'cf_fisd'
    issues: list[str] = []
    for pbs in pbs_dir.glob('*.pbs'):
        text = pbs.read_text()
        if '#PBS -P personal-simerjit' not in text:
            issues.append(f'{pbs.name}: missing -P personal-simerjit')
        if '#PBS -P 43001002' in text or 'aspire2a' in text:
            issues.append(f'{pbs.name}: forbidden billing/queue')
        if '#PBS -q normal' not in text and '#PBS -q ' in text:
            issues.append(f'{pbs.name}: queue not normal')
    return AuditFinding(
        'Rule 55, 56', not issues,
        'PBS scripts: personal-simerjit / -q normal' if not issues else '; '.join(issues),
    )


def check_canonical_row_counts(repo_root: Path) -> AuditFinding:
    verifier = repo_root / 'scripts' / 'cf_fisd' / 'verify_data_integrity.py'
    if not verifier.exists():
        return AuditFinding('Rule 87-89', False, 'verify_data_integrity.py missing')
    text = verifier.read_text()
    issues = []
    for ds, n in CANONICAL_TRAIN_VAL_TEST_TOTAL.items():
        if f"'{ds}': {n}" not in text:
            issues.append(f'{ds}={n} not in verifier')
    return AuditFinding(
        'Rule 87-89', not issues,
        '5/5 canonical row counts encoded' if not issues else '; '.join(issues),
    )


def check_aggregator_amp_parity(repo_root: Path) -> AuditFinding:
    agg = repo_root / 'scripts' / 'cf_fisd' / 'aggregate.py'
    if not agg.exists():
        return AuditFinding('Rule 68, 90', False, 'aggregate.py missing')
    text = agg.read_text()
    has_parity = 'parity skip amp' in text and '_read_toml_amp' in text
    return AuditFinding('Rule 68, 90', has_parity,
                        'aggregator filters config.amp vs toml.amp parity'
                        if has_parity else 'amp parity filter missing')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path,
                    default=Path(__file__).resolve().parent.parent.parent)
    args = ap.parse_args()

    checks = [
        check_pinned_commit,
        check_branch,
        check_no_forbidden_edits,
        check_module_files_namespaced,
        check_state_stamp,
        check_five_datasets,
        check_inheritance_per_dataset,
        check_lambda_zero_off_state,
        check_pbs_billing,
        check_canonical_row_counts,
        check_aggregator_amp_parity,
    ]
    findings = [c(args.repo_root) for c in checks]
    width = max(len(f.rule) for f in findings)
    n_pass = 0
    for f in findings:
        marker = 'PASS' if f.ok else 'FAIL'
        print(f'{marker}  {f.rule:<{width}}  {f.message}')
        n_pass += int(f.ok)
    print(f'\n{n_pass}/{len(findings)} checks passed')
    return 0 if n_pass == len(findings) else 1


if __name__ == '__main__':
    sys.exit(main())
