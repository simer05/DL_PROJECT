#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def mean_std(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return m, s


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-prefix", required=True)
    p.add_argument("--spaces", default="logits,probs,hybrid")
    p.add_argument("--tie-breaks", default="none,val_cross_entropy")
    p.add_argument("--output-md", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    results_dir = Path("results")
    spaces = [x.strip() for x in args.spaces.split(",") if x.strip()]
    tie_breaks = [x.strip() for x in args.tie_breaks.split(",") if x.strip()]

    rows_out: list[dict[str, object]] = []
    for space in spaces:
        for tb in tie_breaks:
            prefix = f"{args.base_prefix}_{space}_{tb}"
            csv_path = results_dir / f"{prefix}_summary.csv"
            if not csv_path.exists():
                continue
            rows = load_csv(csv_path)

            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for r in rows:
                grouped[r["dataset"]].append(r)

            for ds, ds_rows in sorted(grouped.items()):
                base = [float(x["baseline_test"]) for x in ds_rows]
                sel = [float(x["selected_test"]) for x in ds_rows]
                delt = [float(x["test_delta"]) for x in ds_rows]
                ece_b = [float(x["baseline_ece"]) for x in ds_rows]
                ece_s = [float(x["selected_ece"]) for x in ds_rows]
                nll_b = [float(x["baseline_nll"]) for x in ds_rows]
                nll_s = [float(x["selected_nll"]) for x in ds_rows]
                brier_b = [float(x["baseline_brier"]) for x in ds_rows]
                brier_s = [float(x["selected_brier"]) for x in ds_rows]
                eg_b = [float(x["baseline_ensemble_gain"]) for x in ds_rows]
                eg_s = [float(x["selected_ensemble_gain"]) for x in ds_rows]

                bm, bs = mean_std(base)
                sm, ss = mean_std(sel)
                dm, dsd = mean_std(delt)
                eceb, _ = mean_std(ece_b)
                eces, _ = mean_std(ece_s)
                nllb, _ = mean_std(nll_b)
                nlls, _ = mean_std(nll_s)
                bb, _ = mean_std(brier_b)
                bs2, _ = mean_std(brier_s)
                egb, _ = mean_std(eg_b)
                egs, _ = mean_std(eg_s)

                rows_out.append({
                    "space": space,
                    "tie_break": tb,
                    "dataset": ds,
                    "baseline_test_mean": bm,
                    "baseline_test_std": bs,
                    "selected_test_mean": sm,
                    "selected_test_std": ss,
                    "test_delta_mean": dm,
                    "test_delta_std": dsd,
                    "baseline_ece_mean": eceb,
                    "selected_ece_mean": eces,
                    "baseline_nll_mean": nllb,
                    "selected_nll_mean": nlls,
                    "baseline_brier_mean": bb,
                    "selected_brier_mean": bs2,
                    "baseline_ensemble_gain_mean": egb,
                    "selected_ensemble_gain_mean": egs,
                })

    if not rows_out:
        raise SystemExit("No per-space summary CSV files found.")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    ranking = []
    by_space_tb: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows_out:
        by_space_tb[(str(r["space"]), str(r["tie_break"]))].append(r)

    for (space, tb), vals in by_space_tb.items():
        delta_avg = sum(float(v["test_delta_mean"]) for v in vals) / len(vals)
        ece_impr = sum(float(v["baseline_ece_mean"]) - float(v["selected_ece_mean"]) for v in vals) / len(vals)
        nll_impr = sum(float(v["baseline_nll_mean"]) - float(v["selected_nll_mean"]) for v in vals) / len(vals)
        ranking.append({
            "space": space,
            "tie_break": tb,
            "avg_test_delta": delta_avg,
            "avg_ece_improvement": ece_impr,
            "avg_nll_improvement": nll_impr,
            "score": delta_avg + 0.1 * ece_impr + 0.05 * nll_impr,
        })

    ranking.sort(key=lambda x: x["score"], reverse=True)
    recommended = ranking[0]

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps({
        "rows": rows_out,
        "ranking": ranking,
        "recommended": recommended,
    }, indent=2))

    md = []
    md.append("# Dev-only Person B NCL Penalty-space Ablation")
    md.append("")
    md.append("This summary is for dev datasets only (adult, covtype2) and must not be treated as final benchmark evidence.")
    md.append("")
    md.append("## Aggregated Results")
    md.append("")
    md.append("| Space | Tie-break | Dataset | Baseline test (mean±std) | Selected test (mean±std) | Delta | ECE (base->sel) | NLL (base->sel) | Brier (base->sel) | Ensemble gain (base->sel) |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows_out, key=lambda x: (str(x["space"]), str(x["tie_break"]), str(x["dataset"]))):
        md.append(
            f"| {r['space']} | {r['tie_break']} | {r['dataset']} | "
            f"{float(r['baseline_test_mean']):.6f}±{float(r['baseline_test_std']):.6f} | "
            f"{float(r['selected_test_mean']):.6f}±{float(r['selected_test_std']):.6f} | "
            f"{float(r['test_delta_mean']):+.6f} | "
            f"{float(r['baseline_ece_mean']):.6f}->{float(r['selected_ece_mean']):.6f} | "
            f"{float(r['baseline_nll_mean']):.6f}->{float(r['selected_nll_mean']):.6f} | "
            f"{float(r['baseline_brier_mean']):.6f}->{float(r['selected_brier_mean']):.6f} | "
            f"{float(r['baseline_ensemble_gain_mean']):.6f}->{float(r['selected_ensemble_gain_mean']):.6f} |"
        )

    md.append("")
    md.append("## Space Ranking (dev-only)")
    md.append("")
    md.append("| Rank | Space | Tie-break | Avg test delta | Avg ECE improvement | Avg NLL improvement | Composite score |")
    md.append("|---:|---|---|---:|---:|---:|---:|")
    for i, rr in enumerate(ranking, 1):
        md.append(
            f"| {i} | {rr['space']} | {rr['tie_break']} | {rr['avg_test_delta']:+.6f} | "
            f"{rr['avg_ece_improvement']:+.6f} | {rr['avg_nll_improvement']:+.6f} | {rr['score']:+.6f} |"
        )

    md.append("")
    md.append("## Recommended default for TabReD final runs")
    md.append("")
    md.append(
        f"Recommended (dev-only evidence): `ncl_space={recommended['space']}`, tie-break `{recommended['tie_break']}`. "
        "Use validation-selected lambda with `share_training_batches=true`, seeds 0/1/2."
    )
    md.append("")
    md.append("## Final launch command once TabReD is available")
    md.append("")
    md.append("```bash")
    md.append("cd /home/users/ntu/prithvi2/DeepLearning/TabM/paper")
    md.append("python personb/check_tabred_ready.py --strict")
    md.append("qsub personb/personb_tabred_final.pbs")
    md.append("```")

    out_md = Path(args.output_md)
    out_md.write_text("\n".join(md) + "\n")

    print("wrote", out_md)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("recommended", recommended)


if __name__ == "__main__":
    main()
