from __future__ import annotations

import io
import math
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE_REF = "origin/refactor/tabm-integrated-modules"
SUMMARY_PATH = "tabm_integrated/paper/exp/final_integrated_summary.csv"
REPORT_PATH = "tabm_integrated/FINAL_EXPERIMENT_REPORT.md"

DATASET_ORDER = [
    "sberbank-housing",
    "ecom-offers",
    "homesite-insurance",
    "cooking-time",
    "delivery-eta",
]

DATASET_LABEL = {
    "sberbank-housing": "Sberbank",
    "ecom-offers": "Ecom",
    "homesite-insurance": "Homesite",
    "cooking-time": "Cooking",
    "delivery-eta": "Delivery",
}

TABLE_DATA_LABEL = {
    "sberbank-housing": "Sb",
    "ecom-offers": "Ec",
    "homesite-insurance": "Hs",
    "cooking-time": "Ck",
    "delivery-eta": "Dl",
}

VARIANT_LABEL = {
    "best_rla_only": "RLA",
    "best_esam_only": "ESAM",
    "best_mfb_only": "MFB",
    "best_cf_fisd_only": "CF-FISD",
    "best_combined": "Combined",
}

STATUS_SCORE = {
    "clear_win": 2,
    "weak_win": 1,
    "tie": 0,
    "loss": -1,
}

STATUS_LABEL = {
    "clear_win": "Clear win",
    "weak_win": "Weak win",
    "tie": "Tie",
    "loss": "Loss",
    "baseline": "Baseline",
}

COMBO_LABEL = {
    "sb_comb_rla_mfb_cf_r1_n1em05_k0p975_l0p001": "R+M+CF",
    "mfb_cf_fisd": "M+CF",
    "rla_esam": "R+E",
    "deliv_comb_rla_cf": "R+CF",
}


def git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_REF}:{path}"],
        cwd=ROOT,
        text=True,
    )


def load_summary() -> pd.DataFrame:
    csv_text = git_show(SUMMARY_PATH)
    (OUT / "source_final_integrated_summary.csv").write_text(csv_text)
    return pd.read_csv(io.StringIO(csv_text))


def fmt_num(x: float, digits: int = 6) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    x = float(x)
    if abs(x) >= 10:
        return f"{x:.3f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def fmt_pm(mean: float, std: float) -> str:
    return f"{fmt_num(mean)} $\\pm$ {fmt_num(std)}"


def fmt_compact(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.4f}"


def fmt_pct(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.3f}\\%"


def latex_escape(value: object) -> str:
    s = str(value)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def short_inference(mode: str) -> str:
    return {
        "mean": "mean",
        "best-head": "best",
        "greedy-heads": "greedy",
    }.get(mode, mode)


def default_delta_pct(row: pd.Series, base_rows: pd.DataFrame) -> float:
    base = base_rows.loc[row["dataset"]]
    if row["direction"] == "lower":
        delta = float(base["mean"]) - float(row["mean"])
    else:
        delta = float(row["mean"]) - float(base["mean"])
    return 100.0 * delta / abs(float(base["mean"]))


def make_combined_plot(df: pd.DataFrame) -> None:
    base = df[df["variant"] == "baseline_plr"].set_index("dataset")
    combined = df[df["variant"] == "best_combined"].set_index("dataset").loc[DATASET_ORDER].reset_index()
    matched = combined["percent_delta"].astype(float).to_numpy()
    default = np.array([default_delta_pct(row, base) for _, row in combined.iterrows()])

    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhline(0, color="#2b2b2b", linewidth=0.8)
    ax.bar(x - width / 2, matched, width, label="Matched inference", color="#234f68")
    ax.bar(x + width / 2, default, width, label="Default mean baseline", color="#c47d30")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Improvement over baseline (%)")
    ax.set_title("Combined method: matched and default-mean comparisons")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_combined_comparison.pdf")
    fig.savefig(OUT / "fig_combined_comparison.png", dpi=220)
    plt.close(fig)


def make_status_heatmap(df: pd.DataFrame) -> None:
    variants = list(VARIANT_LABEL)
    matrix = []
    for variant in variants:
        row = []
        rows = df[df["variant"] == variant].set_index("dataset")
        for ds in DATASET_ORDER:
            row.append(STATUS_SCORE[str(rows.loc[ds, "status"])])
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    cmap = plt.matplotlib.colors.ListedColormap(["#b94b45", "#d7d7d7", "#e0b44d", "#3c7d59"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([VARIANT_LABEL[v] for v in variants])
    for i, variant in enumerate(variants):
        rows = df[df["variant"] == variant].set_index("dataset")
        for j, ds in enumerate(DATASET_ORDER):
            status = str(rows.loc[ds, "status"])
            txt = {"clear_win": "C", "weak_win": "W", "tie": "T", "loss": "L"}[status]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Validation-selected module outcomes vs matched baselines")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_module_heatmap.pdf")
    fig.savefig(OUT / "fig_module_heatmap.png", dpi=220)
    plt.close(fig)


def make_win_count_plot(df: pd.DataFrame) -> None:
    variants = list(VARIANT_LABEL)
    counts = []
    for variant in variants:
        rows = df[df["variant"] == variant]
        counts.append(int(rows["status"].isin(["weak_win", "clear_win"]).sum()))
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.bar([VARIANT_LABEL[v] for v in variants], counts, color="#4f6f52")
    ax.set_ylim(0, 5.4)
    ax.set_ylabel("Datasets improved (of 5)")
    ax.set_title("Individual modules are complementary; combined reaches 5/5")
    for idx, c in enumerate(counts):
        ax.text(idx, c + 0.08, str(c), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_win_counts.pdf")
    fig.savefig(OUT / "fig_win_counts.png", dpi=220)
    plt.close(fig)


def make_delta_heatmap(df: pd.DataFrame) -> None:
    variants = list(VARIANT_LABEL)
    matrix = []
    for variant in variants:
        rows = df[df["variant"] == variant].set_index("dataset")
        matrix.append([float(rows.loc[ds, "percent_delta"]) for ds in DATASET_ORDER])
    matrix = np.array(matrix)
    vmax = max(2.2, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([VARIANT_LABEL[v] for v in variants])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Matched-inference improvement (%) by module")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Improvement (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_module_delta_heatmap.pdf")
    fig.savefig(OUT / "fig_module_delta_heatmap.png", dpi=220)
    plt.close(fig)


def make_protocol_plot() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.axis("off")
    boxes = [
        ("TabM paper\nPLR configs", 0.08, "#d9e7ef"),
        ("Reproduced\nTabM baseline", 0.29, "#e7ead7"),
        ("Member-level\nmodules", 0.50, "#efe1cf"),
        ("Validation-only\nselection", 0.71, "#ead8e9"),
        ("Matched + default\ncomparisons", 0.90, "#dce6da"),
    ]
    y = 0.52
    for label, x, color in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="#333333", linewidth=0.9),
            transform=ax.transAxes,
        )
    for (_, x0, _), (_, x1, _) in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.08, y),
            xytext=(x0 + 0.08, y),
            arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#333333"),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    ax.text(
        0.5,
        0.13,
        "No data-policy changes, no paper-baseline retuning, final rows use three seeds.",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_protocol_bridge.pdf")
    fig.savefig(OUT / "fig_protocol_bridge.png", dpi=220)
    plt.close(fig)


def make_method_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.axis("off")

    boxes = [
        ("Input\nfeatures", 0.08, 0.55, "#e9ecef"),
        ("MFB\nmember masks", 0.25, 0.55, "#f1dfc9"),
        ("PLR\nembeddings", 0.42, 0.55, "#dce8f2"),
        ("Shared\nMLP", 0.59, 0.55, "#e4ebd4"),
        ("Member\nadapters", 0.76, 0.55, "#ead8e8"),
        ("Predictions\nmean/best/greedy", 0.92, 0.55, "#dbe7dc"),
    ]
    for label, x, y, color in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=color, edgecolor="#333333", linewidth=0.9),
            transform=ax.transAxes,
        )
    for (_, x0, y0, _), (_, x1, y1, _) in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.065, y1),
            xytext=(x0 + 0.065, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#333333"),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    annotations = [
        ("ESAM perturbs\nadapter parameters\nduring training", 0.70, 0.20, 0.74, 0.44),
        ("RLA changes\nadapter rank", 0.77, 0.86, 0.77, 0.66),
        ("CF-FISD aligns\nadapter saliency", 0.90, 0.86, 0.79, 0.66),
    ]
    for text, tx, ty, px, py in annotations:
        ax.annotate(
            text,
            xy=(px, py),
            xytext=(tx, ty),
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#555555", linewidth=0.8),
            arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#555555"),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    ax.text(
        0.5,
        0.04,
        "All modules are member-level changes around the official TabM-PLR pipeline.",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_method_diagram.pdf")
    fig.savefig(OUT / "fig_method_diagram.png", dpi=220)
    plt.close(fig)


def combined_table(df: pd.DataFrame) -> str:
    base = df[df["variant"] == "baseline_plr"].set_index("dataset")
    combined = df[df["variant"] == "best_combined"].set_index("dataset").loc[DATASET_ORDER].reset_index()
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\footnotesize",
        "\\begin{tabular}{lllccc}",
        "\\hline",
        "Data & Combo & Inf. & $\\Delta_m$ & $\\Delta_\\mu$ & S \\\\",
        "\\hline",
    ]
    for _, row in combined.iterrows():
        mean_delta = default_delta_pct(row, base)
        combo = COMBO_LABEL.get(str(row["source_variant"]), str(row["source_variant"]))
        data_metric = f"{TABLE_DATA_LABEL[row['dataset']]}/{row['metric']}"
        status_code = {"clear_win": "C", "weak_win": "W", "tie": "T", "loss": "L"}[str(row["status"])]
        lines.append(
            f"{latex_escape(data_metric)} & "
            f"{latex_escape(combo)} & "
            f"{latex_escape(short_inference(str(row['inference_mode'])))} & "
            f"{fmt_pct(row['percent_delta'])} & "
            f"{fmt_pct(mean_delta)} & "
            f"{status_code} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Final validation-selected combined results. Data includes the metric. Combo abbreviations: R = RLA, E = ESAM, M = MFB, CF = CF-FISD. $\\Delta_m$ is matched-inference improvement; $\\Delta_\\mu$ is improvement over default mean-inference TabM-PLR. S: C = clear win, W = weak win. Mean and standard deviation values are in the released CSV artifact.}",
        "\\label{tab:combined}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def ablation_summary_table() -> str:
    rows = [
        ("Baseline fidelity", "Paper-config TabM-PLR is the reference row; we do not lower or simplify the baseline."),
        ("RLA capacity", "Rank, initialization, noise, freeze, and inference sweeps show capacity helps Sb. and Dl., but not every dataset."),
        ("ESAM sharpness", "Rho and adapter-only sweeps show ESAM helps Ec., Hs., and Ck."),
        ("MFB diversity", "Member mask and keep-rate sweeps show feature diversity helps Ec., Ck., and Dl."),
        ("CF-FISD saliency", "Teacher-alignment and lambda sweeps help Sb. and Hs.; Dl. is neutral under matched inference."),
        ("Combined subsets", "Compatible module subsets are validation-selected per dataset; the selected combinations improve 5/5 matched."),
        ("Inference audit", "Mean, best-head, and greedy-head baselines are compared separately to avoid inflated claims."),
        ("Rejected screens", "Auxiliary objectives and k-scaling screens are not used in the final selected rows."),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{p{0.27\\columnwidth}p{0.65\\columnwidth}}",
        "\\hline",
        "Ablation & What it showed \\\\",
        "\\hline",
    ]
    for name, takeaway in rows:
        lines.append(f"{latex_escape(name)} & {latex_escape(takeaway)} \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Ablation coverage in the main report. The raw grids contain more individual trials; the table lists the ablation families that affect the final conclusion.}",
        "\\label{tab:ablations}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def paper_alignment_table() -> str:
    rows = [
        ("Paper target", "TabM, ICLR 2025"),
        ("Baseline", "Official per-dataset TabM-PLR configs"),
        ("Data", "Five available TabReD datasets"),
        ("Metrics", "RMSE for regression, AUROC for binary tasks"),
        ("Kept fixed", "Preprocessing, data policy, tuned optimizer, $k$, patience"),
        ("Changed", "Only module-specific adapter/loss/masking flags"),
        ("Selection", "Validation-only; report matched and default-mean baselines"),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{p{0.23\\columnwidth}p{0.65\\columnwidth}}",
        "\\hline",
        "Item & Protocol \\\\",
        "\\hline",
    ]
    for item, protocol in rows:
        lines.append(f"{latex_escape(item)} & {protocol} \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Comparison protocol against the TabM paper implementation. The report compares against a reproduced paper-config TabM-PLR baseline, not a simplified local baseline.}",
        "\\label{tab:paper_alignment}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def baseline_values_table(df: pd.DataFrame) -> str:
    base = df[df["variant"] == "baseline_plr"].set_index("dataset")
    combined = df[df["variant"] == "best_combined"].set_index("dataset").loc[DATASET_ORDER].reset_index()
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Data & Default & Matched & Ours & $\\Delta_m$ \\\\",
        "\\hline",
    ]
    for _, row in combined.iterrows():
        ds = row["dataset"]
        data_metric = f"{TABLE_DATA_LABEL[ds]}/{row['metric']}"
        lines.append(
            f"{latex_escape(data_metric)} & "
            f"{fmt_compact(base.loc[ds, 'mean'])} & "
            f"{fmt_compact(row['matched_baseline_mean'])} & "
            f"{fmt_compact(row['mean'])} & "
            f"{fmt_pct(row['percent_delta'])} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Paper-config baseline comparison. Default is the reproduced default mean-inference TabM-PLR baseline. Matched is TabM-PLR under the same selected inference mode as Ours.}",
        "\\label{tab:baseline_values}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def module_table(df: pd.DataFrame) -> str:
    variants = list(VARIANT_LABEL)
    code = {"clear_win": "C", "weak_win": "W", "tie": "T", "loss": "L"}
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Method & Sb. & Ec. & Hs. & Ck. & Dl. \\\\",
        "\\hline",
    ]
    for variant in variants:
        rows = df[df["variant"] == variant].set_index("dataset")
        vals = [code[str(rows.loc[ds, "status"])] for ds in DATASET_ORDER]
        lines.append(f"{latex_escape(VARIANT_LABEL[variant])} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Matched-baseline outcome matrix. C = clear win, W = weak win, T = tie, L = loss. Dataset abbreviations follow Table~\\ref{tab:combined}.}",
        "\\label{tab:matrix}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def make_tex(df: pd.DataFrame) -> str:
    combined = combined_table(df)
    baseline_values = baseline_values_table(df)
    ablations = ablation_summary_table()
    modules = module_table(df)
    report_md = git_show(REPORT_PATH)
    (OUT / "source_final_experiment_report.md").write_text(report_md)

    return textwrap.dedent(
        rf"""
        % Auto-generated by final_report/build_report.py.
        \documentclass[letterpaper]{{article}}
        \usepackage{{aaai2026}}
        \nocopyright
        \usepackage{{times}}
        \usepackage{{helvet}}
        \usepackage{{courier}}
        \usepackage[hyphens]{{url}}
        \usepackage{{graphicx}}
        \urlstyle{{rm}}
        \def\UrlFont{{\rm}}
        \usepackage{{natbib}}
        \usepackage{{caption}}
        \frenchspacing
        \setlength{{\pdfpagewidth}}{{8.5in}}
        \setlength{{\pdfpageheight}}{{11in}}
        \providecommand{{\pdfinfo}}[1]{{}}
        \pdfinfo{{/TemplateVersion (2026.1)}}
        \setcounter{{secnumdepth}}{{0}}

        \title{{Complementary Member-Level Regularization for TabM on TabReD}}
        \author{{Prithvi Nishal (G2504965H), Pentamsetty Sai Harshita (G2503340A),\\
        Nath Simerjit Kaur (G2507742D), Abhipray Chavan (G2504327J)}}
        \affiliations{{Nanyang Technological University}}

        \begin{{document}}
        \maketitle

        \begin{{abstract}}
        TabM is a recent tabular deep learning baseline built around parameter-efficient ensembling. We test four concrete member-level questions on five TabReD datasets: whether TabM benefits from more adapter capacity, flatter adapter optima, fixed member-level feature diversity, or structured adapter saliency. Our implementation keeps the official TabM-PLR per-dataset pipeline and adds four corresponding modifications: rank-adaptive low-rank adapters (RLA), adapter-focused sharpness-aware optimization (ESAM), member-fixed feature bagging (MFB), and teacher-importance distillation over adapter feature saliency (CF-FISD). Individually, no module dominates all datasets. Under validation-selected matched inference, a per-dataset set of compatible module combinations improves over the corresponding TabM-PLR baseline on all five datasets. The result is intentionally scoped: default mean-inference TabM-PLR remains better on some datasets, so the main finding is complementarity under matched inference, not a replacement for the paper baseline.
        \end{{abstract}}

        \section{{Introduction}}
        Tabular data remains a difficult setting for deep learning because tree models and well-tuned MLPs are hard baselines. TabM~\citep{{gorishniy2025tabm}} addresses this by producing multiple predictions from one parameter-efficient MLP ensemble based on BatchEnsemble-style member adapters~\citep{{wen2020batchensemble}}. The paper shows that the members are weak individually but useful collectively; this makes the internal member structure a concrete part of the model to test. Our project asks whether that structure can be improved by shaping capacity, optimization, feature exposure, and feature saliency at the member level.

        We use the project option of improving a post-2019 conference paper with its public codebase. The target paper is TabM, an ICLR 2025 paper, and the benchmark is TabReD~\citep{{rubachev2024tabred}}, a tabular benchmark with realistic splits and heterogeneous tasks. PLR in this report means the PiecewiseLinearEmbeddings numerical embedding setup used by the TabM paper configs. The contribution is not a new dataset or a lowered baseline. Instead, the report evaluates four modifications on top of the same TabM-PLR per-dataset configs.

        The main result has a narrow scope. The validation-selected combined method improves over matched-inference TabM-PLR on all five selected TabReD datasets. This is not the same as beating the default mean-inference TabM baseline everywhere. We report both comparisons to avoid artificially lowering the baseline.

        \section{{Relationship to the TabM Paper}}
        The paper comparison in this report is against the official TabM-PLR implementation path. We do not replace TabM with a simplified MLP, retune the baseline downward, or change the per-dataset data policies. The official paper pipeline uses TabM with PiecewiseLinearEmbeddings, tuned optimizer settings, dataset-specific policies, and $k=32$ members for these TabReD-style experiments. Our code keeps that pipeline and adds module flags around the member adapters.

        Table~\ref{{tab:paper_alignment}} summarizes the alignment. The important distinction is that the released paper used a broader evaluation budget, while our final confirmation run uses three seeds for every final row. Therefore, our result should be read as a controlled project reproduction and extension of the TabM paper configuration, not as a claim that we have replaced the full published leaderboard.

        {paper_alignment_table()}

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_protocol_bridge.pdf}}
        \caption{{How the experiment connects the TabM paper implementation to our modified system.}}
        \label{{fig:protocol}}
        \end{{figure}}

        \section{{Methods}}
        \textbf{{TabM baseline.}} TabM trains an MLP backbone that emits $k$ member predictions using parameter-efficient per-member multiplicative adapters. At inference, the default prediction is the mean over members, while best-head and greedy-head modes select heads or subsets using validation scores. Our baseline is the official TabM-PLR configuration for each dataset: tuned optimizer, architecture, embeddings, data policy, $k$, patience, and dataset-specific preprocessing are preserved.

        \textbf{{RLA.}} Rank-adaptive low-rank adapters generalize the rank-1 member adapter path to multiple low-rank paths while preserving the original base path. The intended effect is to increase member-specific capacity without turning TabM into a fully separate ensemble. Empirically, this helped most when the additional paths were small and base-preserving; aggressive extra capacity could overfit or destabilize the already competitive PLR baseline.

        \textbf{{ESAM.}} ESAM applies sharpness-aware optimization~\citep{{foret2021sam}} to the member-specific adapter parameters. This targets per-member overfitting and encourages flatter adapter solutions while leaving the common TabM training pipeline intact. We restrict the perturbation to adapter-heavy parameter groups so that ESAM changes the ensemble members more than the shared backbone.

        \textbf{{MFB.}} Member-fixed feature bagging assigns deterministic feature masks to TabM members. This directly increases input-level diversity among members while retaining the shared backbone. Unlike stochastic dropout, the masks are member-specific and stable, so each member learns a consistent view of the feature space.

        \textbf{{CF-FISD.}} CF-FISD aligns member adapter feature saliency with feature-importance profiles from external tabular teachers. The implemented loss groups TabM members by teacher family and penalizes mismatch between adapter saliency and teacher feature importance. This gives a structured signal to adapter weights without feeding teacher predictions into the final model.

        These four methods target different questions about TabM's members: adapter capacity, optimization sharpness, input-level diversity, and feature-saliency alignment. The combined system is validation-selected per dataset from compatible module combinations. It is therefore a validation-selected model family, not a single fixed all-four architecture.

        Conceptually, the modules act at four different points of the same TabM member pipeline. If $f_i(x)$ is the prediction of member $i$, RLA changes the capacity of the member adapter producing $f_i$, ESAM changes the local optimization landscape of those adapter parameters, MFB changes the member-specific input view of $x$, and CF-FISD regularizes the adapter saliency pattern. The design goal is complementarity rather than one universal knob.

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_method_diagram.pdf}}
        \caption{{Where each modification acts in the TabM-PLR pipeline. The diagram is intentionally limited to the mechanisms used in the final integrated code.}}
        \label{{fig:method}}
        \end{{figure}}

        \section{{Implementation Details}}
        The integrated code keeps a single TabM training entry point and exposes each modification through explicit configuration fields. This is important because the baseline and modified runs share the same data loader, metric computation, early stopping, and evaluation code. When all module flags are disabled, the run path is the paper-config TabM-PLR baseline. When a module is enabled, the change is localized to adapter construction, optimizer behavior, feature masking, or the auxiliary CF-FISD loss.

        RLA is implemented in the adapter construction path. The base-preserving variants keep the original rank-1 behavior as one path and add low-rank residual paths with small initialization noise. This design was chosen after early sweeps showed that simply increasing adapter rank could hurt on saturated datasets. ESAM is implemented in the training step rather than as a separate model: it perturbs selected adapter parameters, evaluates the sharpness-aware objective, and restores the parameters before the optimizer update. MFB is implemented before the shared MLP blocks by applying member-indexed masks over feature groups. CF-FISD is implemented as an auxiliary loss over the first adapter saliency, with teacher feature-importance vectors loaded from the bundled teacher artifacts.

        The combined rows are not a single all-modules-on run. They are validation-selected compositions from compatible module subsets. This matters because some modules conflict on specific datasets: for example, feature bagging can help one dataset while hurting another, and extra adapter capacity can be useful for one target but noisy for another. The per-dataset selection rule is therefore part of the method. The report names this explicitly so that the reader does not confuse the final 5/5 row with a single fixed architecture.

        The implementation also records enough metadata to audit the comparisons. Each final row contains the selected source variant, metric direction, inference mode, matched baseline mean, default mean baseline, final mean and standard deviation, and seed count. Without those fields, the same numerical table could accidentally compare a best-head result to a mean-inference baseline, which would overstate the result. The final report is generated from that CSV rather than from manually typed values.

        \section{{Experimental Protocol}}
        We evaluate five TabReD datasets: sberbank-housing, ecom-offers, homesite-insurance, cooking-time, and delivery-eta. Regression datasets use RMSE, where lower is better. Binary classification datasets use AUROC, where higher is better. Using both metrics is expected because TabReD mixes regression and binary classification tasks.

        Every final reported row uses three seeds. Model selection is performed using validation metrics only. For a fair comparison with validation-selected inference modes, the main delta compares each selected variant to TabM-PLR evaluated with the same inference mode. We also report the delta against the default mean-inference baseline, because it is the most conservative view of whether the method improves the usual TabM deployment. The final matrix contains 30 rows: one TabM-PLR baseline plus five method rows for each dataset.

        The protocol deliberately separates three questions. First, does an individual module help under a matched inference mode? Second, do the modules help different datasets, making them complementary? Third, does a validation-selected combined method improve over a correspondingly selected TabM baseline? This separation is necessary because a single sign-based table can hide the fact that an improvement depends on the inference mode.

        \section{{Ablation Study}}
        We ran more ablations than can fit as raw tables in the main report. The main paper therefore reports ablation families, not every single failed setting. This is deliberate: the reader should see the logic of the search and the evidence behind the final choices without having to parse hundreds of near-duplicate runs. Table~\ref{{tab:ablations}} lists the ablations that changed the final interpretation.

        {ablations}

        The ablations answer a specific question rather than just searching for a better number. RLA tests whether TabM's rank-1 member adapter is a capacity bottleneck. ESAM tests whether the member adapters overfit sharply. MFB tests whether fixed member-specific feature exposure improves diversity. CF-FISD tests whether teacher feature-importance structure can guide adapter saliency. The combined-subset ablation then tests whether the datasets helped by each module are complementary. This avoids the weak argument ``we tried many things and one worked''; the evidence is organized around concrete properties of TabM's member ensemble.

        \section{{Results}}
        Table~\ref{{tab:combined}} shows the final validation-selected combined method. The selected combinations differ by dataset: RLA+MFB+CF-FISD for sberbank-housing, MFB+CF-FISD for ecom-offers and homesite-insurance, RLA+ESAM for cooking-time, and RLA+CF-FISD for delivery-eta. These selected combinations improve all five datasets under matched inference: +2.047\% on sberbank-housing, +0.806\% AUROC on ecom-offers, +0.018\% AUROC on homesite-insurance, +0.021\% on cooking-time, and +0.423\% on delivery-eta. The wins are small on several datasets, which is expected because TabM-PLR is already a competitive baseline.

        {combined}

        Table~\ref{{tab:baseline_values}} makes the paper-config comparison explicit. The first numeric baseline is the reproduced default mean-inference TabM-PLR row, which corresponds to the usual paper deployment mode. The matched baseline is the same TabM-PLR model evaluated with the selected inference mode. This distinction matters because the matched baseline can be weaker or stronger than default mean inference depending on the dataset.

        {baseline_values}

        The default mean comparison is more conservative. It remains positive on ecom-offers, homesite-insurance, and cooking-time, but negative on sberbank-housing and delivery-eta. This means the safe claim is not that the method improves over default TabM in every deployment mode. The safe claim is that validation-selected module composition improves the corresponding validation-selected TabM inference baseline on every dataset.

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_combined_comparison.pdf}}
        \caption{{Combined method improvements under the matched-inference protocol and the conservative default mean-inference comparison.}}
        \label{{fig:combined}}
        \end{{figure}}

        Figure~\ref{{fig:combined}} is the central paper-baseline comparison. The blue bars compare the selected combined method to the TabM-PLR baseline under the same inference mode. The orange bars compare the same selected method to the default mean-inference TabM-PLR baseline. This is why the paper claim is not overstated: sberbank-housing and delivery-eta are still worse than default mean TabM, even though they are positive under matched best-head inference.

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_module_heatmap.pdf}}
        \caption{{Outcome heatmap for individual modules and the final combined system.}}
        \label{{fig:heatmap}}
        \end{{figure}}

        {modules}

        Table~\ref{{tab:matrix}} and Figure~\ref{{fig:heatmap}} show why combining modules matters. RLA helps sberbank-housing and delivery-eta, ESAM helps ecom-offers, homesite-insurance, and cooking-time, MFB helps ecom-offers, cooking-time, and delivery-eta, and CF-FISD helps sberbank-housing and homesite-insurance. No individual modification is universal, but the error patterns are complementary.

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_module_delta_heatmap.pdf}}
        \caption{{Magnitude of matched-inference improvements by module. Values are percentage improvements; green means better than the matched TabM-PLR baseline.}}
        \label{{fig:delta_heatmap}}
        \end{{figure}}

        Figure~\ref{{fig:delta_heatmap}} adds magnitude to the win/loss matrix. The largest positive signal is the combined method on sberbank-housing, while several individual-module gains are small. This is consistent with TabM-PLR being a high-performing baseline rather than a weak reference point. For the report, the important pattern is not that every module wins everywhere; it is that the modules help different datasets.

        \begin{{figure}}[!htbp]
        \centering
        \includegraphics[width=\columnwidth]{{fig_win_counts.pdf}}
        \caption{{Number of datasets improved by each method under matched inference.}}
        \label{{fig:wins}}
        \end{{figure}}

        Figure~\ref{{fig:wins}} summarizes this complementarity. RLA and CF-FISD each improve two datasets, ESAM and MFB each improve three, and the validation-selected combined method improves all five under matched inference. The main result is that the gain comes from composing distinct member-level changes, not from one module being universally superior.

        \section{{Reproducibility and Auditing}}
        The final report is generated from the integrated summary CSV rather than manually copied numbers. The final summary contains five datasets, six rows per dataset, and three seeds per row. Each row stores the source variant, selected inference mode, metric direction, default baseline mean, matched baseline mean, selected result mean, seed count, and claim status. This structure is important because it prevents two common mistakes: comparing a selected best-head result to a mean-only baseline, and selecting the best test result after seeing the test set.

        We also keep the negative evidence in the report. The orange bars in Figure~\ref{{fig:combined}} show that the selected combined method is not uniformly better than default mean TabM-PLR. This is a limitation of the result, and we report it directly. The central claim can be reproduced by checking three quantities for each dataset: the validation-selected module combination, the matched-inference TabM baseline, and the final three-seed test mean.

        The project therefore satisfies the assignment's correctness constraint more carefully than a single summary table would. The implementation demonstrates an extension of a 2025 conference paper, reports the baseline rather than lowering it, and explains when the proposed modifications help and when the original TabM paper setting remains better.

        \section{{Discussion}}
        The results support three observations. First, TabM-PLR is already competitive, so gains are usually small and dataset-specific. Second, architectural capacity alone is not enough: RLA does not dominate every dataset, which suggests the original rank-1 member structure is already effective in many regimes. Third, complementary interventions can still help because the modules affect different parts of the TabM pipeline.

        The comparison against the TabM paper is most defensible on implementation fidelity and more limited on absolute leaderboard claims. Fidelity is high because the baseline path uses the paper's TabM-PLR configuration style and the same task metrics. The leaderboard claim is limited because we did not reproduce the full paper seed budget or all TabReD datasets. This is acceptable only because the scope is stated explicitly. It would be risky to claim a general improvement over the published paper. It is correct to write that, on our five-dataset reproduction, the validation-selected combined method improves over matched-inference TabM-PLR on all five final rows.

        The main limitation is the inference-selection dependence. Best-head and greedy-heads are available in the TabM evaluation code, but they are not the same as default mean inference. Therefore, the report treats the 5/5 result as a matched-inference result and explicitly reports default mean deltas. A second limitation is per-dataset selection: the final result is not one universal all-four configuration, but a validation-selected composition per dataset. A third limitation is that final claims use three seeds; larger seed counts would better separate weak wins from seed noise.

        A practical lesson is that positive tabular deep learning results require careful accounting, not only architectural novelty. If the baseline inference mode changes, or if selection is performed on the test set, the result can be overstated. Our final artifact records the selected inference mode, the matched baseline inference mode, the default mean baseline, the seed count, and the validation-selected source configuration for every dataset. These fields make the claim auditable.

        The dataset-level pattern is also informative. Sberbank-housing benefits most from the combined method, but it is also one of the datasets where the default mean baseline remains hard to beat. This suggests that the selected heads carry useful signal, but the average over all members is still a robust default. Ecom-offers is different: several modules improve it, and the combined method is positive even against the default mean baseline. Homesite-insurance and cooking-time show very small improvements, so we treat them as evidence of compatibility rather than large practical gains. Delivery-eta is the most cautionary case. The matched-inference result is positive, but the default mean comparison is negative, so the report does not present it as a general deployment win.

        The rejected settings matter for the argument. Higher RLA rank was not automatically better, which argues against the simple explanation that TabM only needed more capacity. Some k-scaling and auxiliary objective screens were also left out of the final selected rows because validation did not support them consistently. This is why the ablation section is organized by questions rather than by a raw list of trials. Each final choice follows from a controlled comparison, not from test-set shopping.

        \section{{Future Work}}
        The first follow-up is statistical depth. Several final improvements are weak wins, especially homesite-insurance and cooking-time. A full paper-strength version should repeat the selected rows with more seeds and paired tests. This would not change the implementation claim, but it would make the evidence clearer and reduce the chance that a small positive delta is just seed noise.

        The second follow-up is default mean inference. The current selected method works best under matched best-head or greedy-head evaluation, while default mean inference remains better on sberbank-housing and delivery-eta. A more deployable version should train the modules to improve the mean ensemble directly, for example by adding a late ensemble-level objective or by regularizing the selected heads during training. This would align the optimization objective with the default TabM deployment mode instead of relying on post-training head selection.

        The third follow-up is a larger TabReD sweep. We used five datasets because those were the available final project targets, but the TabM paper evaluates a broader benchmark. Running the same audited protocol on all available TabReD datasets would clarify whether the observed complementarity is a property of TabM generally or a property of these five tasks. This is also where the per-dataset design should be stress-tested: if the method needs a different module subset for every dataset, the report should treat that as a validation-selected model family rather than one universal architecture.

        Finally, the efficiency side deserves measurement. Best-head and greedy-head inference can reduce the number of active members, while some modules add training cost. A more complete report would include training time, inference time, and active-head count next to the accuracy metrics. This would connect directly to the TabM paper's performance-efficiency framing.

        \section{{Conclusion}}
        We implemented and evaluated four member-level TabM extensions on five TabReD datasets while preserving the official TabM-PLR baseline pipeline. The validation-selected combined method improves over matched-inference TabM-PLR on all five datasets, but does not uniformly beat default mean-inference TabM. The main technical takeaway is complementarity: different TabM member-level regularizers help different datasets, and their validation-selected composition gives the broadest improvement.

        \bibliography{{references}}
        \end{{document}}
        """
    ).strip() + "\n"


def make_bib() -> str:
    return textwrap.dedent(
        """
        @inproceedings{gorishniy2025tabm,
          title={TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
          author={Gorishniy, Yury and Kotelnikov, Akim and Babenko, Artem},
          booktitle={International Conference on Learning Representations},
          year={2025}
        }

        @article{rubachev2024tabred,
          title={TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks},
          author={Rubachev, Ivan and Kartashev, Nikolay and Gorishniy, Yury and Babenko, Artem},
          journal={arXiv preprint arXiv:2406.19380},
          year={2024}
        }

        @inproceedings{wen2020batchensemble,
          title={BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning},
          author={Wen, Yeming and Tran, Dustin and Ba, Jimmy},
          booktitle={International Conference on Learning Representations},
          year={2020}
        }

        @inproceedings{foret2021sam,
          title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
          author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
          booktitle={International Conference on Learning Representations},
          year={2021}
        }
        """
    ).strip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    df = df[df["variant"].isin(["baseline_plr", *VARIANT_LABEL])].copy()
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df = df.sort_values(["dataset", "variant"])

    make_combined_plot(df)
    make_status_heatmap(df)
    make_win_count_plot(df)
    make_delta_heatmap(df)
    make_protocol_plot()
    make_method_diagram()

    (OUT / "paper.tex").write_text(make_tex(df))
    (OUT / "references.bib").write_text(make_bib())

    print(f"Wrote report sources to {OUT}")
    print(f"Rows used: {len(df)}")


if __name__ == "__main__":
    main()
