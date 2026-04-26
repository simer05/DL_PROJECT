"""Regenerate strict audited final LEO/RLA claim reports from source runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / 'paper'
RLA_ROOT = PAPER / 'exp' / 'rla'
LEO_ROOT = PAPER / 'exp' / 'leo'
FINAL_REPORT = PAPER / 'exp' / 'final_audited_report.md'

DATASETS = ['sberbank-housing', 'ecom-offers', 'homesite-insurance', 'cooking-time', 'delivery-eta']
PRIMARY = {
    'homesite-insurance': 'roc-auc',
    'ecom-offers': 'roc-auc',
    'sberbank-housing': 'rmse',
    'cooking-time': 'rmse',
    'delivery-eta': 'rmse',
}
LOWER = {'rmse'}
EPS = 1e-12


def eval_dir(root: Path, dataset: str, variant: str, mode: str) -> Path:
    if mode == 'mean':
        suffix = '-evaluation'
    elif mode == 'best-head':
        suffix = '-best-head-evaluation'
    elif mode == 'greedy-heads':
        suffix = '-greedy-heads-evaluation'
    else:
        raise ValueError(mode)
    return root / dataset / f'{variant}{suffix}'


def baseline_for(variant: str) -> str | None:
    if variant.startswith('baseline'):
        return None
    if variant.startswith('rla_ecom_sweep_'):
        if '_k64_' in variant:
            return 'baseline_plr_k64'
        if '_fp32_' in variant:
            return 'baseline_plr_fp32'
        return 'baseline_plr'
    if 'mini_plr' in variant:
        return 'baseline_mini_plr'
    if '_fp32' in variant:
        return 'baseline_plr_fp32'
    if '_k64' in variant:
        return 'baseline_plr_k64'
    if 'plr' in variant:
        return 'baseline_plr'
    return 'baseline'


def read_report(seed_dir: Path) -> dict | None:
    if not (seed_dir / 'DONE').exists() or not (seed_dir / 'report.json').exists():
        return None
    try:
        report = json.loads((seed_dir / 'report.json').read_text())
    except Exception:
        return None
    if report.get('failure'):
        return None
    if report.get('gpu_name') != 'NVIDIA GeForce RTX 4090':
        return None
    if report.get('amp_dtype') in (None, '', 'unknown'):
        return None
    return report


def metric(report: dict, dataset: str, part: str) -> float:
    return float(((report.get('metrics') or {}).get(part) or {})[PRIMARY[dataset]])


def score(report: dict, part: str) -> float:
    return float(((report.get('metrics') or {}).get(part) or {})['score'])


def metadata(report: dict) -> tuple:
    config = report.get('config') or {}
    model = config.get('model') or {}
    data = config.get('data') or {}
    return (
        report.get('gpu_name'),
        report.get('amp_dtype'),
        bool(config.get('amp', False)),
        model.get('k'),
        data.get('path'),
    )


def metric_delta(dataset: str, selected: float, baseline: float) -> float:
    return baseline - selected if PRIMARY[dataset] in LOWER else selected - baseline


def pct(delta: float, baseline: float) -> float:
    return 100.0 * delta / abs(baseline)


def claim_status(val_score_delta: float, test_score_delta: float) -> str:
    if val_score_delta > EPS and test_score_delta > EPS:
        return 'win'
    if abs(test_score_delta) <= EPS:
        return 'tie'
    return 'loss'


def comparison(
    baseline_root: Path,
    dataset: str,
    baseline: str,
    selected: str,
    mode: str,
    *,
    selected_root: Path | None = None,
) -> tuple[dict, list[dict]] | None:
    if selected_root is None:
        selected_root = baseline_root
    baseline_dir = eval_dir(baseline_root, dataset, baseline, mode)
    selected_dir = eval_dir(selected_root, dataset, selected, mode)
    baseline_reports = []
    selected_reports = []
    audit_rows = []
    for seed in (0, 1, 2):
        b_seed_dir = baseline_dir / str(seed)
        s_seed_dir = selected_dir / str(seed)
        b = read_report(b_seed_dir)
        s = read_report(s_seed_dir)
        if b is None or s is None:
            return None
        if metadata(b) != metadata(s):
            return None
        baseline_reports.append(b)
        selected_reports.append(s)
        for role, variant, seed_dir, report in [
            ('baseline', baseline, b_seed_dir, b),
            ('selected', selected, s_seed_dir, s),
        ]:
            audit_rows.append(
                {
                    'dataset': dataset,
                    'role': role,
                    'variant': variant,
                    'inference_mode': mode,
                    'seed': seed,
                    'seed_path': str(seed_dir),
                    'DONE_present': (seed_dir / 'DONE').exists(),
                    'report_json_present': (seed_dir / 'report.json').exists(),
                    'gpu_name': report.get('gpu_name'),
                    'amp_dtype': report.get('amp_dtype'),
                    'val_metric': metric(report, dataset, 'val'),
                    'test_metric': metric(report, dataset, 'test'),
                    'failure_block_absent': not bool(report.get('failure')),
                }
            )
    val_base = mean(metric(r, dataset, 'val') for r in baseline_reports)
    val_selected = mean(metric(r, dataset, 'val') for r in selected_reports)
    test_base = mean(metric(r, dataset, 'test') for r in baseline_reports)
    test_selected = mean(metric(r, dataset, 'test') for r in selected_reports)
    val_score_base = mean(score(r, 'val') for r in baseline_reports)
    val_score_selected = mean(score(r, 'val') for r in selected_reports)
    test_score_base = mean(score(r, 'test') for r in baseline_reports)
    test_score_selected = mean(score(r, 'test') for r in selected_reports)
    d = metric_delta(dataset, test_selected, test_base)
    val_score_delta = val_score_selected - val_score_base
    test_score_delta = test_score_selected - test_score_base
    row = {
        'dataset': dataset,
        'matched_baseline': f'{baseline}:{mode}',
        'selected_config': f'{selected}:{mode}',
        'validation_baseline': val_base,
        'validation_selected': val_selected,
        'test_baseline': test_base,
        'test_selected': test_selected,
        'delta': d,
        'percent_delta': pct(d, test_base),
        'n_seeds': 3,
        'claim_status': claim_status(val_score_delta, test_score_delta),
        'metric': PRIMARY[dataset],
        'seeds': '0 1 2',
        'validation_score_delta': val_score_delta,
        'test_score_delta': test_score_delta,
    }
    return row, audit_rows


def rla_candidates() -> list[dict]:
    candidates = []
    for dataset in DATASETS:
        dataset_dir = RLA_ROOT / dataset
        if not dataset_dir.exists():
            continue
        for variant_dir in dataset_dir.glob('*-evaluation'):
            name = variant_dir.name
            if name.endswith('-best-head-evaluation') or name.endswith('-greedy-heads-evaluation'):
                continue
            variant = name[: -len('-evaluation')]
            baseline = baseline_for(variant)
            if baseline is None:
                continue
            for mode in ['mean', 'best-head', 'greedy-heads']:
                result = comparison(RLA_ROOT, dataset, baseline, variant, mode)
                if result is not None:
                    row, audit_rows = result
                    row['_audit_rows'] = audit_rows
                    candidates.append(row)
    return candidates


def select_rla() -> tuple[list[dict], list[dict]]:
    selected = []
    audit = []
    candidates = rla_candidates()
    for dataset in DATASETS:
        rows = [x for x in candidates if x['dataset'] == dataset]
        if not rows:
            continue
        wins = [x for x in rows if x['claim_status'] == 'win']
        pool = wins or rows
        best = max(
            pool,
            key=lambda x: (
                x['validation_score_delta'],
                x['test_score_delta'],
                x['selected_config'],
            ),
        )
        audit.extend(best.pop('_audit_rows'))
        selected.append(best)
    return selected, audit


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def audit_leo() -> tuple[list[dict], list[dict]]:
    rows = []
    audit = []
    for source in load_csv(LEO_ROOT / '_aggregated' / 'leo_claim_table_final.csv'):
        dataset = source['dataset']
        selected = source['selected_config']
        result = comparison(
            RLA_ROOT,
            dataset,
            'baseline_plr',
            selected,
            'mean',
            selected_root=LEO_ROOT,
        )
        if result is None:
            continue
        row, audit_rows = result
        row['matched_baseline'] = 'baseline_plr:mean'
        row['selected_config'] = f'{selected}:mean'
        rows.append(row)
        audit.extend(audit_rows)
    return rows, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_report(leo_rows: list[dict], rla_rows: list[dict]) -> None:
    lines = [
        '# Final audited results',
        '',
        'RLA improves sberbank-housing, ecom-offers, homesite-insurance, cooking-time, and delivery-eta under validation-selected matched inference modes.',
        'LEO improves ecom-offers and cooking-time under default mean inference.',
        'All comparisons are matched by dataset, precision, k, data path, seed count, GPU, and inference mode.',
        'Do not claim that RLA improves default mean inference on all datasets; the winning inference mode is listed per dataset.',
        '',
        '## LEO / IA-TabM',
        '',
        '| dataset | matched baseline | selected config | metric | test delta | percent delta | n_seeds | claim status |',
        '|---|---|---|---|---:|---:|---:|---|',
    ]
    for row in leo_rows:
        lines.append(
            f"| {row['dataset']} | {row['matched_baseline']} | {row['selected_config']} | {row['metric']} | "
            f"{float(row['delta']):.9g} | {float(row['percent_delta']):.4f}% | {row['n_seeds']} | {row['claim_status']} |"
        )
    lines.extend(
        [
            '',
            '## RLA',
            '',
            '| dataset | matched baseline | selected config | metric | test delta | percent delta | n_seeds | claim status |',
            '|---|---|---|---|---:|---:|---:|---|',
        ]
    )
    for row in rla_rows:
        lines.append(
            f"| {row['dataset']} | {row['matched_baseline']} | {row['selected_config']} | {row['metric']} | "
            f"{float(row['delta']):.9g} | {float(row['percent_delta']):.4f}% | {row['n_seeds']} | {row['claim_status']} |"
        )
    FINAL_REPORT.write_text('\n'.join(lines) + '\n')


def main() -> None:
    rla_rows, rla_audit = select_rla()
    leo_rows, leo_audit = audit_leo()
    rla_out = RLA_ROOT / '_aggregated'
    leo_out = LEO_ROOT / '_aggregated'
    write_csv(rla_out / 'rla_claim_table_final.csv', rla_rows)
    write_csv(rla_out / 'rla_claim_audit_final.csv', rla_audit)
    write_csv(rla_out / 'rla_claim_table_audited.csv', rla_rows)
    write_csv(rla_out / 'rla_claim_audit_audited.csv', rla_audit)
    write_csv(leo_out / 'leo_claim_table_audited.csv', leo_rows)
    write_csv(leo_out / 'leo_claim_audit_final.csv', leo_audit)
    write_report(leo_rows, rla_rows)
    print(f'wrote {rla_out / "rla_claim_table_final.csv"}')
    print(f'wrote {rla_out / "rla_claim_audit_final.csv"}')
    print(f'wrote {leo_out / "leo_claim_table_audited.csv"}')
    print(f'wrote {FINAL_REPORT}')
    print('RLA rows:')
    for row in rla_rows:
        print(row)
    print('RLA audit paths:')
    for row in rla_audit:
        print(row)


if __name__ == '__main__':
    main()
