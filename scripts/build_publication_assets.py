from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from build_full_benchmark_report import (
    PLOT_TASK_ORDER,
    TASK_LABELS,
    _build_segment_rows,
    _load_summaries,
)


MODEL_LABELS = {
    "gemma2:9b": "Gemma2 9B",
    "aya-expanse:8b": "Aya Expanse 8B",
    "llama3.1:8b-instruct": "Llama 3.1 8B Instruct",
    "qwen2.5:7b-instruct": "Qwen2.5 7B Instruct",
    "falcon3:7b-instruct": "Falcon3 7B Instruct",
}

SEGMENT_LABELS = {
    "Intent Classification": "Intent",
    "Legal Classification": "Legal",
    "Machine Translation (ENG)": "MT-ENG",
    "Machine Translation (FAS)": "MT-FAS",
    "Machine Translation (JPN)": "MT-JPN",
    "Summarization": "Summ.",
}

PRIMARY_METRIC_LABELS = {
    "exact_match": "Exact Match",
    "macro_f1": "Macro-F1",
    "wer_vs_reference": "WER",
    "chrf": "chrF",
    "bertscore_f1": "BERTScore-F1",
}

APPENDIX_TASK_ORDER = [
    "gec",
    "intent_classification",
    "legal_classification",
    "machine_translation",
    "ner",
    "pos",
    "summarization",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-ready tables and figures for a benchmark run.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing benchmark summaries and report CSVs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution for raster output.",
    )
    return parser.parse_args()


def _pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _pretty_segment(segment: str) -> str:
    return SEGMENT_LABELS.get(segment, segment)


def _pretty_metric(metric: str) -> str:
    return PRIMARY_METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _sample_caption(summary_frames: dict[str, pd.DataFrame]) -> str:
    parts: list[str] = []
    for task in ["gec", "intent_classification", "legal_classification", "machine_translation", "ner", "pos", "summarization"]:
        if task not in summary_frames:
            continue
        df = summary_frames[task]
        if task == "machine_translation" and "target_lang" in df.columns:
            lang_sizes = []
            mt_sizes = df[["target_lang", "samples"]].drop_duplicates().sort_values("target_lang")
            for _, row in mt_sizes.iterrows():
                lang_sizes.append(f"{str(row['target_lang']).upper()}={int(row['samples'])}")
            parts.append(f"MT({', '.join(lang_sizes)})")
        else:
            label = TASK_LABELS.get(task, task).replace(" Classification", "")
            parts.append(f"{label}={int(df['samples'].iloc[0])}")
    return ", ".join(parts)


def _write_overall_table(article_dir: Path, overall_table: pd.DataFrame, winner_table: pd.DataFrame) -> None:
    wins = winner_table["winner"].value_counts()
    latex_table = overall_table.copy()
    latex_table["Model"] = latex_table["model"].map(_pretty_model)
    latex_table["Rank"] = latex_table["rank"].astype(int)
    latex_table["Avg. Quality"] = latex_table["avg_normalized_quality"].map(lambda value: f"{value:.1f}")
    latex_table["Avg. Latency (s)"] = latex_table["avg_latency_seconds"].map(lambda value: f"{value:.2f}")
    latex_table["Segment Wins"] = latex_table["model"].map(lambda model: int(wins.get(model, 0)))
    latex_table = latex_table[["Model", "Rank", "Avg. Quality", "Avg. Latency (s)", "Segment Wins"]]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Overall ranking of the evaluated models across the nine benchmark segments in the deterministic capped benchmark run. Normalized quality is averaged over segment-level, direction-aware scores.}",
        r"\label{tab:capped-overall-ranking}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Rank & Avg. Quality & Avg. Latency (s) & Segment Wins \\",
        r"\midrule",
    ]
    for _, row in latex_table.iterrows():
        lines.append(
            f"{row['Model']} & {row['Rank']} & {row['Avg. Quality']} & {row['Avg. Latency (s)']} & {row['Segment Wins']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (article_dir / "overall_ranking_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_winners_table(article_dir: Path, winner_table: pd.DataFrame, sample_caption: str) -> None:
    latex_table = winner_table.copy()
    latex_table["Task Segment"] = latex_table["task_segment"].map(_pretty_segment)
    latex_table["Winner"] = latex_table["winner"].map(_pretty_model)
    latex_table["Primary Metric"] = latex_table["primary_metric"].map(_pretty_metric)
    latex_table["Value"] = latex_table["winner_value"].map(lambda value: f"{value:.3f}")
    latex_table = latex_table[["Task Segment", "Winner", "Primary Metric", "Value"]]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best-performing model per benchmark segment on the deterministic capped evaluation run ({sample_caption}). Higher Exact Match, Macro-F1, chrF, and BERTScore-F1 are better.}}",
        r"\label{tab:capped-task-winners}",
        r"\begin{tabular}{lllr}",
        r"\toprule",
        r"Task Segment & Winner & Primary Metric & Value \\",
        r"\midrule",
    ]
    for _, row in latex_table.iterrows():
        lines.append(
            f"{row['Task Segment']} & {row['Winner']} & {row['Primary Metric']} & {row['Value']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (article_dir / "task_winners_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_figure_snippets(article_dir: Path) -> None:
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=0.92\textwidth]{article_assets/figures/model_quality_heatmap.pdf}",
        r"\caption{Normalized quality heatmap across the nine benchmark segments. Scores are normalized within each segment so that higher is always better.}",
        r"\label{fig:model-quality-heatmap}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=0.82\textwidth]{article_assets/figures/quality_vs_latency.pdf}",
        r"\caption{Quality-latency trade-off across models and task families. Each point denotes one model-task pair.}",
        r"\label{fig:quality-vs-latency}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=0.92\textwidth]{article_assets/figures/task_winners.pdf}",
        r"\caption{Normalized quality margin between the winning model and the runner-up for each benchmark segment.}",
        r"\label{fig:task-winner-margins}",
        r"\end{figure*}",
    ]
    (article_dir / "figure_snippets.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def _primary_metric_for_task(task_name: str) -> tuple[str, bool]:
    if task_name == "machine_translation":
        return ("chrf", False)
    if task_name == "summarization":
        return ("bertscore_f1", False)
    if task_name == "gec":
        return ("exact_match", False)
    return ("macro_f1", False)


def _task_table_columns(task_name: str, df: pd.DataFrame) -> list[tuple[str, str]]:
    if task_name == "gec":
        pairs = [("model", "Model"), ("exact_match", "Exact Match"), ("wer_vs_reference", "WER"), ("avg_latency_seconds", "Latency (s)")]
    elif task_name in {"intent_classification", "legal_classification"}:
        pairs = [("model", "Model"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("avg_latency_seconds", "Latency (s)")]
    elif task_name == "machine_translation":
        pairs = [("target_lang", "Lang"), ("model", "Model"), ("wer_vs_reference", "WER"), ("bleu", "BLEU"), ("chrf", "chrF"), ("avg_latency_seconds", "Latency (s)")]
    elif task_name == "ner":
        pairs = [("model", "Model"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("entity_f1", "Entity F1"), ("avg_latency_seconds", "Latency (s)")]
    elif task_name == "pos":
        pairs = [("model", "Model"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("avg_latency_seconds", "Latency (s)")]
    else:
        pairs = [
            ("model", "Model"),
            ("wer_vs_reference", "WER"),
            ("rouge_1", "ROUGE-1"),
            ("rouge_l", "ROUGE-L"),
            ("bertscore_f1", "BERTScore-F1"),
            ("avg_latency_seconds", "Latency (s)"),
        ]
    return [(column, label) for column, label in pairs if column in df.columns]


def _format_latex_value(value: object, column: str) -> str:
    if pd.isna(value):
        return "--"
    if column == "model":
        return _pretty_model(str(value))
    if column == "target_lang":
        return str(value).upper()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _write_appendix_task_table(task_name: str, df: pd.DataFrame, tables_dir: Path) -> None:
    metric, lower_is_better = _primary_metric_for_task(task_name)
    sort_columns = [metric, "avg_latency_seconds"] if "avg_latency_seconds" in df.columns else [metric]
    sort_ascending = [lower_is_better, True] if "avg_latency_seconds" in df.columns else [lower_is_better]
    working = df.sort_values(sort_columns, ascending=sort_ascending).copy()
    columns = _task_table_columns(task_name, working)
    align = "".join("l" if column in {"model", "target_lang"} else "r" for column, _ in columns)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Per-model results for {_pretty_segment(TASK_LABELS.get(task_name, task_name.replace('_', ' ').title()))}.}}",
        rf"\label{{tab:appendix-{task_name.replace('_', '-')}-analysis}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(label for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in working.iterrows():
        lines.append(" & ".join(_format_latex_value(row[column], column) for column, _ in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / f"{task_name}_analysis_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def _plot_appendix_task_figure(task_name: str, df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    metric, lower_is_better = _primary_metric_for_task(task_name)
    sns.set_theme(style="whitegrid", context="paper")

    if task_name == "machine_translation":
        target_langs = [lang for lang in ["eng", "fas", "jpn"] if lang in set(df["target_lang"])]
        fig, axes = plt.subplots(1, len(target_langs), figsize=(14.0, 4.6), sharey=False)
        if len(target_langs) == 1:
            axes = [axes]
        for ax, target_lang in zip(axes, target_langs, strict=False):
            lang_df = df[df["target_lang"] == target_lang].sort_values(metric, ascending=lower_is_better).copy()
            labels = [_pretty_model(model) for model in lang_df["model"]]
            values = lang_df[metric].astype(float).tolist()
            bars = ax.barh(labels, values, color=sns.color_palette("crest", n_colors=len(lang_df)), edgecolor="#17324d")
            ax.set_title(str(target_lang).upper())
            ax.set_xlabel(_pretty_metric(metric))
            if lower_is_better:
                ax.invert_yaxis()
            for bar, latency in zip(bars, lang_df["avg_latency_seconds"], strict=False):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {latency:.2f}s", va="center", ha="left", fontsize=7.5)
        fig.suptitle("Machine Translation by Target Language", fontsize=14, y=1.02)
        plt.tight_layout()
        _save_dual(fig, figures_dir / "machine_translation_analysis", dpi)
        return

    working = df.sort_values(metric, ascending=lower_is_better).copy()
    labels = [_pretty_model(model) for model in working["model"]]
    values = working[metric].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bars = ax.barh(labels, values, color=sns.color_palette("crest", n_colors=len(working)), edgecolor="#17324d")
    ax.set_title(f"{TASK_LABELS.get(task_name, task_name.title())} Analysis", fontsize=14, pad=10)
    ax.set_xlabel(_pretty_metric(metric))
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    if lower_is_better:
        ax.invert_yaxis()
    for bar, latency in zip(bars, working["avg_latency_seconds"], strict=False):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {latency:.2f}s", va="center", ha="left", fontsize=7.5)
    plt.tight_layout()
    _save_dual(fig, figures_dir / f"{task_name}_analysis", dpi)


def _write_appendix_task_analysis(summary_frames: dict[str, pd.DataFrame], article_dir: Path, dpi: int) -> None:
    tables_dir = article_dir / "appendix_tables"
    figures_dir = article_dir / "appendix_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    section_lines = [r"\section{Per-Task Analysis}", r"\label{sec:appendix-per-task-analysis}", ""]
    for task_name in APPENDIX_TASK_ORDER:
        if task_name not in summary_frames:
            continue
        df = summary_frames[task_name].copy()
        _write_appendix_task_table(task_name, df, tables_dir)
        _plot_appendix_task_figure(task_name, df, figures_dir, dpi)

        section_lines.extend(
            [
                rf"\subsection{{{TASK_LABELS.get(task_name, task_name.replace('_', ' ').title())}}}",
                rf"\input{{\assetsdir/appendix_tables/{task_name}_analysis_table.tex}}",
                r"\begin{figure*}[t]",
                r"\centering",
                rf"\includegraphics[width=0.92\textwidth]{{\assetsdir/appendix_figures/{task_name}_analysis.pdf}}",
                rf"\caption{{Per-task comparison for {TASK_LABELS.get(task_name, task_name.replace('_', ' ').title())}. Bars show the primary metric and annotations show average latency.}}",
                rf"\label{{fig:appendix-{task_name.replace('_', '-')}-analysis}}",
                r"\end{figure*}",
                "",
            ]
        )
    (article_dir / "appendix_task_analysis.tex").write_text("\n".join(section_lines) + "\n", encoding="ascii")


def _save_dual(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.parent / f"{output_base.name}_{dpi}dpi.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_task_winners_publication(winner_table: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    plot_df = winner_table.copy()
    plot_df["winner_display"] = plot_df["winner"].map(_pretty_model)
    plot_df["task_segment"] = plot_df["task_segment"].map(_pretty_segment)
    ordered = [_pretty_segment(segment) for segment in PLOT_TASK_ORDER]
    plot_df["task_segment"] = pd.Categorical(plot_df["task_segment"], categories=ordered, ordered=True)
    plot_df = plot_df.sort_values("task_segment")

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    palette = sns.color_palette("crest", n_colors=len(plot_df))
    bars = ax.barh(plot_df["task_segment"], plot_df["quality_margin"], color=palette, edgecolor="#17324d", linewidth=0.8)
    ax.set_title("Winner Margin Over Runner-up", fontsize=15, pad=12)
    ax.set_xlabel("Normalized quality margin (points)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_xlim(0, max(5, float(plot_df["quality_margin"].max()) * 1.18))

    for bar, winner_name, margin in zip(bars, plot_df["winner_display"], plot_df["quality_margin"], strict=False):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{winner_name} (+{margin:.1f})",
            va="center",
            ha="left",
            fontsize=8.4,
        )

    plt.tight_layout()
    _save_dual(fig, figures_dir / "task_winners", dpi)


def _plot_quality_heatmap_publication(combined: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    heatmap = combined.pivot_table(index="model", columns="segment_label", values="quality_score", aggfunc="mean")
    ordered_columns = [column for column in PLOT_TASK_ORDER if column in heatmap.columns]
    heatmap = heatmap.reindex(columns=ordered_columns)
    heatmap.index = [_pretty_model(index) for index in heatmap.index]
    heatmap.columns = [_pretty_segment(column) for column in heatmap.columns]
    heatmap = heatmap.loc[heatmap.mean(axis=1).sort_values(ascending=False).index]

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Normalized quality (0-100)"},
        ax=ax,
    )
    ax.set_title("Model Quality Heatmap", fontsize=15, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=22, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_dual(fig, figures_dir / "model_quality_heatmap", dpi)


def _plot_quality_latency_publication(combined: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    scatter = (
        combined.groupby(["model", "task"], as_index=False)
        .agg(
            avg_quality_score=("quality_score", "mean"),
            avg_latency_seconds=("avg_latency_seconds", "mean"),
        )
    )
    scatter["task_label"] = scatter["task"].map(lambda task: TASK_LABELS[task])

    models = list(scatter["model"].drop_duplicates())
    palette = dict(zip(models, sns.color_palette("tab10", n_colors=len(models)), strict=False))
    markers = {
        "GEC": "o",
        "Intent Classification": "s",
        "Legal Classification": "X",
        "Machine Translation": "D",
        "NER": "^",
        "POS": "P",
        "Summarization": "v",
    }

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for _, row in scatter.iterrows():
        ax.scatter(
            row["avg_latency_seconds"],
            row["avg_quality_score"],
            s=115,
            marker=markers.get(row["task_label"], "o"),
            color=palette[row["model"]],
            edgecolor="#16324f",
            linewidth=0.7,
            alpha=0.92,
        )
        ax.text(
            row["avg_latency_seconds"] + 0.03,
            row["avg_quality_score"] + 0.8,
            _pretty_model(row["model"]),
            fontsize=7.6,
            alpha=0.92,
        )

    ax.set_title("Quality vs Latency Trade-off", fontsize=15, pad=12)
    ax.set_xlabel("Average latency (seconds)")
    ax.set_ylabel("Average normalized quality")
    ax.grid(True, linestyle="--", alpha=0.25)

    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[model], markeredgecolor="#16324f", markersize=7, label=_pretty_model(model))
        for model in models
    ]
    task_handles = [
        Line2D([0], [0], marker=marker, color="#1f2933", linestyle="", markersize=7, label=_pretty_segment(task_label))
        for task_label, marker in markers.items()
    ]
    legend_models = ax.legend(handles=model_handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.add_artist(legend_models)
    ax.legend(handles=task_handles, title="Task family", bbox_to_anchor=(1.02, 0.46), loc="upper left", frameon=True)

    plt.tight_layout()
    _save_dual(fig, figures_dir / "quality_vs_latency", dpi)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    article_dir = results_dir / "article_assets"
    figures_dir = article_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_frames, _ = _load_summaries(results_dir)
    combined, _ = _build_segment_rows(summary_frames)
    overall_table = pd.read_csv(results_dir / "report_model_overall.csv")
    winner_table = pd.read_csv(results_dir / "report_task_winners.csv")
    sample_caption = _sample_caption(summary_frames)

    _write_overall_table(article_dir, overall_table, winner_table)
    _write_winners_table(article_dir, winner_table, sample_caption)
    _write_figure_snippets(article_dir)
    _write_appendix_task_analysis(summary_frames, article_dir, args.dpi)
    _plot_task_winners_publication(winner_table, figures_dir, args.dpi)
    _plot_quality_heatmap_publication(combined, figures_dir, args.dpi)
    _plot_quality_latency_publication(combined, figures_dir, args.dpi)

    print(f"Saved LaTeX tables to {article_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
