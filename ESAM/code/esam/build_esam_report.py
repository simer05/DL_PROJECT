#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _relative_improvement(row: pd.Series) -> float | None:
    b = row.get('baseline_test_score')
    s = row.get('test_score')
    if b is None or s is None:
        return None
    denom = abs(float(b)) if abs(float(b)) > 1e-12 else None
    if denom is None:
        return None
    return float((float(s) - float(b)) / denom)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--tag', type=str, default='safe_esam')
    args = p.parse_args()

    payload = json.loads(args.selection.read_text())
    all_rows = pd.DataFrame(payload.get('all_rows', []))
    selected = pd.DataFrame(payload.get('selected_rows', []))
    if all_rows.empty:
        raise SystemExit('No rows in selection payload')

    args.out_root.mkdir(parents=True, exist_ok=True)
    tables = args.out_root / 'tables'
    tables.mkdir(parents=True, exist_ok=True)

    if not selected.empty:
        selected['relative_improvement_vs_baseline'] = selected.apply(_relative_improvement, axis=1)
        selected['win_loss_neutral'] = selected['test_delta_vs_baseline'].apply(
            lambda x: 'win' if x > 0 else ('loss' if x < 0 else 'neutral')
        )

    all_rows.to_csv(tables / f'{args.tag}_all_rows.csv', index=False)
    selected.to_csv(tables / f'{args.tag}_selected_rows.csv', index=False)

    by_variant = (
        all_rows.groupby(['dataset', 'train_fraction', 'variant'])
        .agg(val_score=('val_score', 'mean'), test_score=('test_score', 'mean'), n=('seed', 'count'))
        .reset_index()
    )
    by_variant.to_csv(tables / f'{args.tag}_by_variant.csv', index=False)

    md: list[str] = []
    md.append(f'# SAFE ESAM Report ({args.tag})')
    md.append('')
    md.append('## Protocol')
    md.append(f"- Selection: {payload.get('protocol', {}).get('selection', 'unknown')}")
    md.append(f"- Margin: {payload.get('protocol', {}).get('selection_margin', 'n/a')}")
    md.append('')
    md.append('## Selected Rows')
    if selected.empty:
        md.append('- No selected rows available.')
    else:
        md.append(selected.to_markdown(index=False))

    md.append('')
    md.append('## Variant Summary')
    md.append(by_variant.to_markdown(index=False))

    report_path = args.out_root / f'SAFE_ESAM_REPORT_{args.tag}.md'
    report_path.write_text('\n'.join(md))
    if args.tag == 'final':
        (args.out_root / 'SAFE_ESAM_FINAL_REPORT.md').write_text('\n'.join(md))
    print(f'[done] {report_path}')


if __name__ == '__main__':
    main()
