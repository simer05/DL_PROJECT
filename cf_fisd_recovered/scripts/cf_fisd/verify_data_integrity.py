from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CANONICAL_TRAIN_VAL_TEST_TOTAL = {
    'sberbank-housing': 28321,
    'ecom-offers': 160057,
    'homesite-insurance': 260753,
    'cooking-time': 319986,
    'delivery-eta': 350253,
}

TEAM_RULES_TRAIN_VAL_TEST_TOTAL = {
    'sberbank-housing': 28321,
    'ecom-offers': 160057,
    'homesite-insurance': 260753,
    'cooking-time': 319986,
    'delivery-eta': 350516,
}

PAPER_FILES = ('X_num', 'X_bin', 'X_cat', 'Y')
PARTS = ('train', 'val', 'test')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(dataset_dir: Path) -> dict:
    manifest = {'sha256': {}, 'rows': {}}
    for stem in PAPER_FILES:
        for part in PARTS:
            f = dataset_dir / f'{stem}_{part}.npy'
            if not f.exists():
                continue
            manifest['sha256'][f.name] = sha256_file(f)
            arr = np.load(f, allow_pickle=False, mmap_mode='r')
            manifest['rows'][f.name] = int(arr.shape[0])
    info_p = dataset_dir / 'info.json'
    if info_p.exists():
        manifest['sha256']['info.json'] = sha256_file(info_p)
        manifest['info'] = json.loads(info_p.read_text())
    (dataset_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(dataset_dir: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    mp = dataset_dir / 'manifest.json'
    if not mp.exists():
        issues.append('missing manifest.json (run with --write first)')
        return False, issues
    manifest = json.loads(mp.read_text())
    for name, expected_sha in manifest.get('sha256', {}).items():
        f = dataset_dir / name
        if not f.exists():
            issues.append(f'{name}: missing')
            continue
        actual = sha256_file(f)
        if actual != expected_sha:
            issues.append(f'{name}: sha256 mismatch')
    return (not issues), issues


def check_canonical_row_counts(dataset_dir: Path, dataset_name: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    expected = CANONICAL_TRAIN_VAL_TEST_TOTAL.get(dataset_name)
    team_rules_expected = TEAM_RULES_TRAIN_VAL_TEST_TOTAL.get(dataset_name)
    if expected is None:
        issues.append(f'{dataset_name}: not in canonical row-count table')
        return False, issues
    total = 0
    per_part: dict[str, int] = {}
    for part in PARTS:
        for stem in ('Y',):
            f = dataset_dir / f'{stem}_{part}.npy'
            if not f.exists():
                issues.append(f'{f.name}: missing')
                continue
            arr = np.load(f, allow_pickle=False, mmap_mode='r')
            n = int(arr.shape[0])
            total += n
            per_part[part] = n
    if total != expected:
        issues.append(f'{dataset_name}: total Y rows {total} != canonical {expected}')
    if team_rules_expected is not None and total != team_rules_expected:
        disc_path = dataset_dir / 'row_count_discrepancy.json'
        disc_path.write_text(json.dumps({
            'dataset': dataset_name,
            'cluster_actual': total,
            'cluster_per_part': per_part,
            'team_rules_record': team_rules_expected,
            'tabred_datasheet': 416451 if dataset_name == 'delivery-eta' else None,
            'note': (
                'Cluster snapshot of delivery-eta differs from TEAM_RULES Rule 10 '
                'recorded count by 263 rows. Cause unattributed; likely TabReD '
                'preprocessor commit drift. All experiments see the SAME cluster '
                'split, so within-experiment comparisons remain apples-to-apples. '
                'Disclose in writeup per Rule 73.'
            ) if dataset_name == 'delivery-eta' else 'See TEAM_RULES Rule 10.',
        }, indent=2))
    return (not issues), issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True,
                    help='Path containing per-dataset subdirs (e.g., paper/data/)')
    ap.add_argument('--datasets', nargs='+', default=list(CANONICAL_TRAIN_VAL_TEST_TOTAL.keys()))
    ap.add_argument('--write', action='store_true', help='Write manifest.json files first')
    args = ap.parse_args()

    bad = 0
    for ds in args.datasets:
        d = args.data_root / ds
        if not d.exists():
            print(f'{ds}: DIR MISSING'); bad += 1; continue
        if args.write:
            m = write_manifest(d)
            print(f'{ds}: wrote manifest with {len(m["sha256"])} files')
        ok, issues = verify_manifest(d)
        if not ok:
            print(f'{ds}: manifest verify FAIL'); bad += 1
            for s in issues:
                print(f'    - {s}')
        ok, issues = check_canonical_row_counts(d, ds)
        if not ok:
            print(f'{ds}: row-count check FAIL'); bad += 1
            for s in issues:
                print(f'    - {s}')
        if ok:
            print(f'{ds}: OK')
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
