from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

MODULE_FILES = (
    'paper/lib/cf_fisd.py',
    'paper/bin/cf_fisd_teachers.py',
    'paper/bin/model.py',
    'paper/test_cf_fisd.py',
    'paper/tools/generate_cf_fisd_configs.py',
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    repo_root: Path = args.repo_root.resolve()
    out: Path = (args.out or (repo_root / 'paper' / '.cf_fisd_state.json')).resolve()

    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    branch = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=repo_root, capture_output=True, text=True, check=False,
    ).stdout.strip()

    file_shas: dict[str, str] = {}
    for rel in MODULE_FILES:
        p = repo_root / rel
        file_shas[rel] = sha256_file(p) if p.exists() else 'MISSING'

    stamp = {
        'module': 'cf_fisd',
        'upstream_commit': head,
        'branch': branch,
        'dirty': bool(dirty),
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'module_file_shas': file_shas,
    }
    out.write_text(json.dumps(stamp, indent=2))
    print(f'wrote {out}')
    for k, v in stamp.items():
        if k != 'module_file_shas':
            print(f'  {k}: {v}')
    print('  module_file_shas:')
    for k, v in file_shas.items():
        print(f'    {k}: {v[:12]}')


if __name__ == '__main__':
    main()
