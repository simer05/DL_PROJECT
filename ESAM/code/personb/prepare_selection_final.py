#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_csv(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _is_classification(metrics_val: dict) -> bool:
    return "cross-entropy" in metrics_val or "roc-auc" in metrics_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-based selection for Person B final runs")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-prefix", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True, help="comma-separated dataset keys")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--lambdas", type=str, default="0,5e-4,1e-3,2e-3")
    parser.add_argument("--ncl-space", type=str, default="logits", choices=["logits", "probs", "hybrid"])
    parser.add_argument("--ncl-warmup-epochs", type=int, default=10)
    parser.add_argument("--require-share-training-batches", action="store_true")
    parser.add_argument(
        "--tie-break",
        type=str,
        default="none",
        choices=["none", "val_cross_entropy"],
        help="For classification tasks, break val-score ties with lower val cross-entropy",
    )
    parser.add_argument("--score-tie-eps", type=float, default=1e-12)
    args = parser.parse_args()

    datasets = parse_csv_str(args.datasets)
    seeds = [int(x) for x in parse_csv_str(args.seeds)]
    lambdas = parse_float_csv(args.lambdas)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    for report in sorted(args.run_root.glob("*/seed*/lambda_*/report.json")):
        data = json.loads(report.read_text())
        cfg = data["config"]
        ds = report.parent.parent.parent.name
        seed = int(cfg["seed"])
        lam = float(cfg.get("lambda_ncl", 0.0))
        warm = int(cfg.get("ncl_warmup_epochs", 0))
        space = cfg.get("ncl_space", "logits")
        share_batches = bool(cfg.get("model", {}).get("share_training_batches", True))

        row = {
            "dataset": ds,
            "seed": seed,
            "lambda_ncl": lam,
            "warmup": warm,
            "space": space,
            "share_training_batches": share_batches,
            "val_score": float(data["metrics"]["val"]["score"]),
            "val_metrics": data["metrics"]["val"],
            "test_score": float(data["metrics"]["test"]["score"]),
            "run_dir": str(report.parent),
            "report_path": str(report),
        }
        rows.append(row)

    filtered = [
        r
        for r in rows
        if r["dataset"] in datasets
        and r["seed"] in seeds
        and r["warmup"] == args.ncl_warmup_epochs
        and r["space"] == args.ncl_space
        and any(abs(r["lambda_ncl"] - x) < 1e-12 for x in lambdas)
    ]
    if args.require_share_training_batches:
        filtered = [r for r in filtered if r["share_training_batches"]]

    selected = []
    missing = []

    for ds in datasets:
        for sd in seeds:
            cand = [r for r in filtered if r["dataset"] == ds and r["seed"] == sd]
            have = sorted({r["lambda_ncl"] for r in cand})
            for lam in lambdas:
                if not any(abs(x - lam) < 1e-12 for x in have):
                    missing.append({"dataset": ds, "seed": sd, "lambda_ncl": lam})

            if not cand:
                continue

            cand.sort(key=lambda x: x["lambda_ncl"])
            best = cand[0]
            for c in cand[1:]:
                better = c["val_score"] > best["val_score"] + args.score_tie_eps
                tie = abs(c["val_score"] - best["val_score"]) <= args.score_tie_eps
                if better:
                    best = c
                    continue
                if tie and args.tie_break == "val_cross_entropy":
                    if _is_classification(c["val_metrics"]) and _is_classification(best["val_metrics"]):
                        c_ce = float(c["val_metrics"].get("cross-entropy", float("inf")))
                        b_ce = float(best["val_metrics"].get("cross-entropy", float("inf")))
                        if c_ce < b_ce:
                            best = c

            base = next((r for r in cand if abs(r["lambda_ncl"]) < 1e-12), None)
            if base is None:
                missing.append({"dataset": ds, "seed": sd, "lambda_ncl": 0.0})
                continue

            selected.append(
                {
                    "dataset": ds,
                    "seed": sd,
                    "baseline_run": base["run_dir"],
                    "baseline_val": base["val_score"],
                    "baseline_test": base["test_score"],
                    "selected_lambda": best["lambda_ncl"],
                    "selected_run": best["run_dir"],
                    "selected_val": best["val_score"],
                    "selected_test": best["test_score"],
                    "test_delta": best["test_score"] - base["test_score"],
                    "share_training_batches": best["share_training_batches"],
                }
            )

    out = {
        "protocol": {
            "run_root": str(args.run_root),
            "datasets": datasets,
            "seeds": seeds,
            "lambdas": lambdas,
            "ncl_space": args.ncl_space,
            "ncl_warmup_epochs": args.ncl_warmup_epochs,
            "require_share_training_batches": args.require_share_training_batches,
            "tie_break": args.tie_break,
            "score_tie_eps": args.score_tie_eps,
        },
        "missing_expected_runs": missing,
        "selected_by_val": selected,
    }

    out_path = out_dir / f"{args.result_prefix}_selected_runs.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)
    if missing:
        print("missing runs:", len(missing))


if __name__ == "__main__":
    main()
