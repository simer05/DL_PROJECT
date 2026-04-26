"""Verify every non-marker file in cf_fisd_recovered/ has size > 0 and contains non-NUL bytes.
Skips DONE marker files (intentionally zero bytes per paper convention).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(r'C:\Users\Simerjit Kaur\cf_fisd_recovered')
ALLOWED_EMPTY = {'DONE'}  # filenames that are legitimately zero bytes

def check(p: Path) -> tuple[int, int]:
    size = p.stat().st_size
    if size == 0: return 0, 0
    with open(p, 'rb') as f:
        data = f.read(min(size, 1_000_000))
    nonzero = sum(1 for b in data if b != 0)
    return size, nonzero

bad = []
total = 0
total_skipped = 0
for p in sorted(ROOT.rglob('*')):
    if not p.is_file(): continue
    if p.name in ALLOWED_EMPTY:
        total_skipped += 1; continue
    total += 1
    sz, nz = check(p)
    if sz == 0 or nz == 0:
        rel = p.relative_to(ROOT)
        print(f'CORRUPT  {rel}  size={sz}  non-NUL bytes in first MB={nz}')
        bad.append(str(rel))

print(f'\n{total} non-marker files checked, {total_skipped} marker files skipped, {len(bad)} corrupted')
if bad:
    print('\nRe-pull these from cluster via SFTP:')
    for b in bad: print(f'  {b}')
sys.exit(1 if bad else 0)
