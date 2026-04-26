#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def iter_reports(root: Path):
    for report in sorted(root.glob("*/seed*/lambda_*/report.json")):
        yield report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in iter_reports(args.outputs):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        diversity = data.get("diversity", {})

        dataset = config.get("data", {}).get("path", "").split("/")[-1]
        seed = config.get("seed")
        lam = config.get("lambda_ncl", 0.0)
        use_ncl = config.get("use_ncl", False)

        row = {
            "dataset": dataset,
            "seed": seed,
            "use_ncl": use_ncl,
            "lambda_ncl": lam,
            "val_score": metrics.get("val", {}).get("score"),
            "test_score": metrics.get("test", {}).get("score"),
            "val_mean_centered_corr": diversity.get("val", {}).get("mean_centered_corr"),
            "val_pairwise_disagreement": diversity.get("val", {}).get("mean_pairwise_disagreement"),
            "val_member_std": diversity.get("val", {}).get("member_std"),
            "time": data.get("time"),
            "report": str(path),
        }
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# Person B NCL Results", "", "| dataset | seed | lambda_ncl | use_ncl | val_score | test_score | val_mean_centered_corr | val_pairwise_disagreement | val_member_std |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['seed']} | {r['lambda_ncl']} | {r['use_ncl']} | {r['val_score']} | {r['test_score']} | {r['val_mean_centered_corr']} | {r['val_pairwise_disagreement']} | {r['val_member_std']} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
