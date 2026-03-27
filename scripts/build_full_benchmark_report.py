from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


PRIMARY_METRIC_BY_TASK = {
    "gec": ("exact_match", False),
    "intent_classification": ("macro_f1", False),
    "legal_classification": ("macro_f1", False),
    "machine_translation": ("wer_vs_reference", True),
    "ner": ("macro_f1", False),
    "pos": ("macro_f1", False),
    "summarization": ("wer_vs_reference", True),
}

TASK_LABELS = {
    "gec": "GEC",
    "intent_classification": "Intent Classification",
    "legal_classification": "Legal Classification",
    "machine_translation": "Machine Translation",
    "ner": "NER",
    "pos": "POS",
    "summarization": "Summarization",
}

PLOT_TASK_ORDER = [
    "GEC",
    "Intent Classification",
    "Legal Classification",
    "Machine Translation (ENG)",
    "Machine Translation (FAS)",
    "Machine Translation (JPN)",
    "NER",
    "POS",
    "Summarization",
]


@dataclass(frozen=True)
class SegmentSummary:
    task: str
    label: str
    primary_metric: str
    lower_is_better: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined report for a full benchmark suite run.")
    parser.add_argument(
        "--results-dir",
        default="results/full_benchmark_suite",
        help="Directory containing *_summary.csv outputs from a full benchmark run.",
    )
    return parser.parse_args()


def _segment_label(task: str, row: pd.Series) -> str:
    if task == "machine_translation":
        return f"{TASK_LABELS[task]} ({str(row['target_lang']).upper()})"
    return TASK_LABELS.get(task, task.replace("_", " ").title())


def _task_from_path(path: Path) -> str:
    return path.name.replace("_summary.csv", "")


def _load_summaries(results_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    summary_paths = sorted(results_dir.glob("*_summary.csv"))
    summary_frames: dict[str, pd.DataFrame] = {}

    for path in summary_paths:
        task = _task_from_path(path)
        df = pd.read_csv(path)
        if "task" not in df.columns:
            df["task"] = task
        summary_frames[task] = df

    log_path = results_dir / "tmux_full_benchmark.log"
    failed_tasks: list[str] = []
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        task_match = re.search(r"Tasks:\s*\[(.*?)\]", text)
        if task_match:
            requested_tasks = [item.strip().strip("'\"") for item in task_match.group(1).split(",")]
            completed_tasks = list(summary_frames.keys())
            failed_tasks = [task for task in requested_tasks if task not in completed_tasks]

    return summary_frames, failed_tasks


def _build_segment_rows(summary_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[SegmentSummary]]:
    rows: list[pd.DataFrame] = []
    segment_summaries: list[SegmentSummary] = []

    for task, df in summary_frames.items():
        if task not in PRIMARY_METRIC_BY_TASK:
            continue
        primary_metric, lower_is_better = PRIMARY_METRIC_BY_TASK[task]
        working = df.copy()
        if task == "machine_translation":
            group_columns = ["target_lang"]
        else:
            group_columns = [None]

        for group in group_columns:
            if group is None:
                segment_df = working.copy()
                label = TASK_LABELS[task]
            else:
                for target_lang, target_df in working.groupby(group, sort=False):
                    label = f"{TASK_LABELS[task]} ({str(target_lang).upper()})"
                    annotated = target_df.copy()
                    annotated["segment_label"] = label
                    annotated["primary_metric"] = primary_metric
                    annotated["lower_is_better"] = lower_is_better
                    rows.append(annotated)
                    segment_summaries.append(SegmentSummary(task=task, label=label, primary_metric=primary_metric, lower_is_better=lower_is_better))
                continue

            segment_df["segment_label"] = label
            segment_df["primary_metric"] = primary_metric
            segment_df["lower_is_better"] = lower_is_better
            rows.append(segment_df)
            segment_summaries.append(SegmentSummary(task=task, label=label, primary_metric=primary_metric, lower_is_better=lower_is_better))

    combined = pd.concat(rows, ignore_index=True)
    combined["quality_score"] = np.nan

    for segment in segment_summaries:
        mask = combined["segment_label"] == segment.label
        values = combined.loc[mask, segment.primary_metric].astype(float)
        if values.nunique(dropna=True) <= 1:
            combined.loc[mask, "quality_score"] = 50.0
            continue
        if segment.lower_is_better:
            scaled = 100.0 * (values.max() - values) / (values.max() - values.min())
        else:
            scaled = 100.0 * (values - values.min()) / (values.max() - values.min())
        combined.loc[mask, "quality_score"] = scaled

    return combined, segment_summaries


def _winner_table(combined: pd.DataFrame, segment_summaries: list[SegmentSummary]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in segment_summaries:
        segment_df = combined.loc[combined["segment_label"] == segment.label].copy()
        ascending = segment.lower_is_better
        ranked = segment_df.sort_values(by=[segment.primary_metric, "avg_latency_seconds"], ascending=[ascending, True]).reset_index(drop=True)

        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        fastest = segment_df.sort_values("avg_latency_seconds", ascending=True).iloc[0]

        winner_value = float(winner[segment.primary_metric])
        runner_value = float(runner_up[segment.primary_metric]) if runner_up is not None else np.nan
        winner_quality_score = float(winner["quality_score"])
        runner_up_quality_score = float(runner_up["quality_score"]) if runner_up is not None else np.nan
        if pd.isna(runner_value):
            margin = np.nan
        else:
            margin = (runner_value - winner_value) if segment.lower_is_better else (winner_value - runner_value)

        note = ""
        if segment.task == "legal_classification" and float(segment_df[segment.primary_metric].fillna(0).max()) == 0.0:
            note = "All models scored 0, likely evaluation or prompting issue."
        elif winner["model"] == fastest["model"]:
            note = "Winner also had the fastest average latency."
        elif winner["avg_latency_seconds"] <= fastest["avg_latency_seconds"] * 1.1:
            note = "Winner was within 10% of the fastest model."

        rows.append(
            {
                "task_segment": segment.label,
                "primary_metric": segment.primary_metric,
                "winner": winner["model"],
                "winner_value": winner_value,
                "winner_quality_score": winner_quality_score,
                "runner_up": runner_up["model"] if runner_up is not None else "",
                "runner_up_value": runner_value,
                "runner_up_quality_score": runner_up_quality_score,
                "quality_margin": (winner_quality_score - runner_up_quality_score) if not pd.isna(runner_up_quality_score) else np.nan,
                "margin": margin,
                "fastest_model": fastest["model"],
                "fastest_latency_seconds": float(fastest["avg_latency_seconds"]),
                "samples": int(winner["samples"]) if "samples" in winner else np.nan,
                "note": note,
            }
        )

    table = pd.DataFrame(rows)
    return table


def _overall_model_table(combined: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        combined.groupby("model", as_index=False)
        .agg(
            tasks_completed=("segment_label", "nunique"),
            avg_normalized_quality=("quality_score", "mean"),
            median_normalized_quality=("quality_score", "median"),
            avg_latency_seconds=("avg_latency_seconds", "mean"),
        )
        .sort_values(["avg_normalized_quality", "avg_latency_seconds"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped = grouped[["rank", "model", "tasks_completed", "avg_normalized_quality", "median_normalized_quality", "avg_latency_seconds"]]
    return grouped


def _save_csvs(results_dir: Path, winner_table: pd.DataFrame, overall_model_table: pd.DataFrame, combined: pd.DataFrame) -> None:
    winner_table.to_csv(results_dir / "report_task_winners.csv", index=False)
    overall_model_table.to_csv(results_dir / "report_model_overall.csv", index=False)

    detailed_columns = [
        column
        for column in [
            "task",
            "segment_label",
            "model",
            "samples",
            "target_lang",
            "exact_match",
            "accuracy",
            "macro_f1",
            "wer_vs_reference",
            "cer_vs_reference",
            "avg_latency_seconds",
            "quality_score",
        ]
        if column in combined.columns
    ]
    combined[detailed_columns].sort_values(["segment_label", "quality_score"], ascending=[True, False]).to_csv(
        results_dir / "report_detailed_results.csv",
        index=False,
    )


def _plot_task_winners(winner_table: pd.DataFrame, charts_dir: Path) -> None:
    plot_df = winner_table.copy()
    plot_df["winner_display"] = plot_df["winner"] + "\n" + plot_df["quality_margin"].map(lambda value: f"+{value:.1f} pts")
    plot_df["task_segment"] = pd.Categorical(plot_df["task_segment"], categories=PLOT_TASK_ORDER, ordered=True)
    plot_df = plot_df.sort_values("task_segment")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    palette = sns.color_palette("crest", n_colors=len(plot_df))
    bars = ax.barh(plot_df["task_segment"], plot_df["quality_margin"], color=palette, edgecolor="#17324d")
    ax.set_title("Winner Margin Over Runner-up", fontsize=16, pad=16)
    ax.set_xlabel("Normalized quality margin (points)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_xlim(0, max(5, plot_df["quality_margin"].max() * 1.15))

    for bar, label in zip(bars, plot_df["winner_display"], strict=False):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {label}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(charts_dir / "task_winners.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_quality_heatmap(combined: pd.DataFrame, charts_dir: Path) -> None:
    heatmap = combined.pivot_table(index="model", columns="segment_label", values="quality_score", aggfunc="mean")
    ordered_columns = [column for column in PLOT_TASK_ORDER if column in heatmap.columns]
    heatmap = heatmap.reindex(columns=ordered_columns)
    heatmap = heatmap.loc[heatmap.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
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
    ax.set_title("Model Quality Heatmap Across Completed Tasks", fontsize=16, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(charts_dir / "model_quality_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_quality_latency(combined: pd.DataFrame, charts_dir: Path) -> None:
    scatter = (
        combined.groupby(["model", "task"], as_index=False)
        .agg(
            avg_quality_score=("quality_score", "mean"),
            avg_latency_seconds=("avg_latency_seconds", "mean"),
        )
    )
    task_display = {task: TASK_LABELS[task] for task in scatter["task"].unique()}
    scatter["task_label"] = scatter["task"].map(task_display)

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

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for _, row in scatter.iterrows():
        ax.scatter(
            row["avg_latency_seconds"],
            row["avg_quality_score"],
            s=160,
            marker=markers.get(row["task_label"], "o"),
            color=palette[row["model"]],
            edgecolor="#16324f",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.text(
            row["avg_latency_seconds"] + 0.03,
            row["avg_quality_score"] + 0.6,
            row["model"],
            fontsize=8.5,
            alpha=0.9,
        )

    ax.set_title("Quality vs Latency Trade-off", fontsize=16, pad=14)
    ax.set_xlabel("Average latency (seconds)")
    ax.set_ylabel("Average normalized quality")
    ax.grid(True, linestyle="--", alpha=0.25)

    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[model], markeredgecolor="#16324f", markersize=8, label=model)
        for model in models
    ]
    task_handles = [
        Line2D([0], [0], marker=marker, color="#1f2933", linestyle="", markersize=8, label=task_label)
        for task_label, marker in markers.items()
    ]
    legend_models = ax.legend(handles=model_handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.add_artist(legend_models)
    ax.legend(handles=task_handles, title="Task", bbox_to_anchor=(1.02, 0.48), loc="upper left")

    plt.tight_layout()
    fig.savefig(charts_dir / "quality_vs_latency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    results_dir: Path,
    winner_table: pd.DataFrame,
    overall_model_table: pd.DataFrame,
    failed_tasks: list[str],
) -> None:
    completed_segments = winner_table["task_segment"].tolist()
    lines: list[str] = []
    lines.append("# Full Benchmark Report")
    lines.append("")
    lines.append(f"This report summarizes the benchmark run captured in `{results_dir}/`.")
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append("- Completed task segments: " + ", ".join(f"`{segment}`" for segment in completed_segments))
    if failed_tasks:
        lines.append("- Incomplete tasks: " + ", ".join(f"`{task}`" for task in failed_tasks))
    else:
        lines.append("- Incomplete tasks: none")
    lines.append("")
    lines.append("## Overall Model Ranking")
    lines.append("")
    lines.append(_dataframe_to_markdown(overall_model_table))
    lines.append("")
    lines.append("## Best Model Per Task Segment")
    lines.append("")
    lines.append(_dataframe_to_markdown(winner_table))
    lines.append("")
    lines.append("## Diagrams")
    lines.append("")
    lines.append("![Best model by task segment](charts/task_winners.png)")
    lines.append("")
    lines.append("![Model quality heatmap](charts/model_quality_heatmap.png)")
    lines.append("")
    lines.append("![Quality versus latency](charts/quality_vs_latency.png)")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    top_model = overall_model_table.iloc[0]["model"] if not overall_model_table.empty else "n/a"
    lines.append(f"- `{top_model}` ranks first overall on the normalized quality aggregate for this run.")
    if "Legal Classification" in completed_segments:
        lines.append("- Legal classification is now coarse-grained (`Volume N` labels), which avoids the previous all-zero opaque-ID setup, though the task remains difficult.")
    if "POS" in completed_segments:
        lines.append("- POS tagging completed after switching the UD loader to a parser that tolerates `_` head values in the CoNLL-U files.")
    if "Summarization" in completed_segments:
        lines.append("- Summarization remains the slowest task family in this sample and is currently scored with edit-distance metrics in the summary table.")
    lines.append("")

    (results_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[_format_markdown_value(value) for value in row] for row in df.itertuples(index=False, name=None)]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_frames, failed_tasks = _load_summaries(results_dir)
    combined, segment_summaries = _build_segment_rows(summary_frames)
    winner_table = _winner_table(combined, segment_summaries)
    overall_model_table = _overall_model_table(combined)

    _save_csvs(results_dir, winner_table, overall_model_table, combined)
    _plot_task_winners(winner_table, charts_dir)
    _plot_quality_heatmap(combined, charts_dir)
    _plot_quality_latency(combined, charts_dir)
    _write_markdown_report(results_dir, winner_table, overall_model_table, failed_tasks)

    print(f"Saved report to {results_dir / 'REPORT.md'}")
    print(f"Saved charts to {charts_dir}")


if __name__ == "__main__":
    main()
