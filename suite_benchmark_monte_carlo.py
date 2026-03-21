from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from benchmark_suite import GenerationConfig, list_tasks, run_task, save_run_outputs

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct",
    "aya-expanse:8b",
    "llama3.1:8b",
]

ALL_TASKS = "all"
LOWER_IS_BETTER_SUFFIXES = ("wer", "cer", "latency", "error")


def parse_args() -> argparse.Namespace:
    task_choices = [ALL_TASKS, *list_tasks()]
    parser = argparse.ArgumentParser(description="Run repeated benchmark tasks and report mean and SEM.")
    parser.add_argument(
        "--task",
        default=ALL_TASKS,
        choices=task_choices,
        help="Benchmark task to run, or 'all' to execute every supported task.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model names to benchmark.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="How many examples to score per repeat. Use 0 to run on the full test dataset.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="How many repeated runs to perform per task.")
    parser.add_argument("--random-state", type=int, default=42, help="Base sampling seed.")
    parser.add_argument("--data-csv", default="data.csv", help="Path to the benchmark dataset registry CSV.")
    parser.add_argument("--output-dir", default="results/suite_monte_carlo", help="Where to save benchmark outputs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to Ollama.")
    parser.add_argument("--num-predict", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-request timeout.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing repeat outputs when present instead of recomputing them.",
    )
    return parser.parse_args()


def _selected_tasks(task_name: str) -> list[str]:
    if task_name == ALL_TASKS:
        return list_tasks()
    return [task_name]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if not math.isnan(value) else "nan"
    return str(value)


def _aggregate_repeated_summaries(summary_by_repeat: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated_summary = pd.concat(summary_by_repeat, ignore_index=True)
    key_columns = [column for column in ["task", "model"] if column in repeated_summary.columns]
    numeric_columns = [
        column
        for column in repeated_summary.columns
        if column not in key_columns + ["repeat"] and pd.api.types.is_numeric_dtype(repeated_summary[column])
    ]

    aggregated = repeated_summary[key_columns].drop_duplicates().sort_values(key_columns).reset_index(drop=True)
    grouped = repeated_summary.groupby(key_columns, sort=False)

    for column in numeric_columns:
        aggregated[f"{column}_mean"] = grouped[column].mean().to_numpy()
        aggregated[f"{column}_sem"] = grouped[column].sem(ddof=1).fillna(0.0).to_numpy()

    return aggregated, repeated_summary


def _load_repeat_outputs(repeat_output_dir: Path, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(repeat_output_dir / f"{task_name}_summary.csv")
    raw = pd.read_csv(repeat_output_dir / f"{task_name}_predictions.csv")
    return summary, raw


def _metric_preferences(summary: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "exact_match_mean",
            "accuracy_mean",
            "macro_f1_mean",
            "wer_vs_reference_mean",
            "cer_vs_reference_mean",
            "avg_latency_seconds_mean",
            "exact_match",
            "accuracy",
            "macro_f1",
            "wer_vs_reference",
            "cer_vs_reference",
            "avg_latency_seconds",
        ]
        if column in summary.columns
    ]


def _is_lower_better(metric_name: str) -> bool:
    return any(token in metric_name for token in LOWER_IS_BETTER_SUFFIXES)


def _print_summary(summary: pd.DataFrame) -> None:
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not math.isnan(x) else "nan"))


def _build_performance_by_task(task_summaries: dict[str, pd.DataFrame], selected_tasks: list[str]) -> pd.DataFrame:
    performance_rows: list[dict[str, object]] = []

    for task_name in selected_tasks:
        summary = task_summaries[task_name].copy()
        metric_options = _metric_preferences(summary)
        if not metric_options:
            continue

        metric = metric_options[0]
        ascending = _is_lower_better(metric)
        ranked = summary.sort_values(metric, ascending=ascending).reset_index(drop=True)
        best_row = ranked.iloc[0]
        sem_column = metric.replace("_mean", "_sem")

        performance_rows.append(
            {
                "task": task_name,
                "primary_metric": metric,
                "best_model": best_row["model"],
                "best_score_mean": best_row[metric],
                "best_score_sem": best_row[sem_column] if sem_column in best_row.index else 0.0,
            }
        )

    return pd.DataFrame(performance_rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    sample_size = None if args.sample_size <= 0 else args.sample_size
    selected_tasks = _selected_tasks(args.task)
    config = GenerationConfig(
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout_seconds=args.timeout_seconds,
    )

    task_summaries: dict[str, pd.DataFrame] = {}
    combined_summaries: list[pd.DataFrame] = []

    for task_name in selected_tasks:
        print(f"\n=== Running task: {task_name} ===")
        repeat_summaries: list[pd.DataFrame] = []
        repeat_predictions: list[pd.DataFrame] = []

        for repeat_index in range(args.repeats):
            repeat_number = repeat_index + 1
            repeat_seed = args.random_state + repeat_index
            repeat_output_dir = output_root / task_name / f"repeat_{repeat_number:02d}"

            if args.resume and (repeat_output_dir / f"{task_name}_summary.csv").exists():
                print(f"Reusing repeat {repeat_number}/{args.repeats} from {repeat_output_dir}")
                summary, raw = _load_repeat_outputs(repeat_output_dir, task_name)
            else:
                print(f"Running repeat {repeat_number}/{args.repeats} with seed {repeat_seed}")
                summary, raw = run_task(
                    task_name=task_name,
                    models=args.models,
                    sample_size=sample_size,
                    random_state=repeat_seed,
                    data_csv=args.data_csv,
                    config=config,
                )
                save_run_outputs(summary, raw, repeat_output_dir, task_name)

            summary = summary.copy()
            raw = raw.copy()
            summary["repeat"] = repeat_number
            raw["repeat"] = repeat_number
            repeat_summaries.append(summary)
            repeat_predictions.append(raw)

        aggregated_summary, repeated_summary = _aggregate_repeated_summaries(repeat_summaries)
        repeated_predictions = pd.concat(repeat_predictions, ignore_index=True)
        task_output_dir = output_root / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        aggregated_summary_path = task_output_dir / f"{task_name}_summary_with_sem.csv"
        repeated_summary_path = task_output_dir / f"{task_name}_repeat_summaries.csv"
        repeated_predictions_path = task_output_dir / f"{task_name}_repeat_predictions.csv"

        aggregated_summary.to_csv(aggregated_summary_path, index=False)
        repeated_summary.to_csv(repeated_summary_path, index=False)
        repeated_predictions.to_csv(repeated_predictions_path, index=False)

        task_summaries[task_name] = aggregated_summary
        combined_summaries.append(aggregated_summary)

        _print_summary(aggregated_summary)
        print(f"\nSaved aggregated summary CSV: {aggregated_summary_path}")
        print(f"Saved repeat summaries CSV: {repeated_summary_path}")
        print(f"Saved repeat predictions CSV: {repeated_predictions_path}")

    combined_summary = pd.concat(combined_summaries, ignore_index=True)
    combined_summary_path = output_root / "all_tasks_summary_with_sem.csv"
    combined_summary.to_csv(combined_summary_path, index=False)
    print(f"\nSaved combined summary CSV: {combined_summary_path}")

    performance_by_task = _build_performance_by_task(task_summaries, selected_tasks)
    if not performance_by_task.empty:
        performance_by_task_path = output_root / "performance_by_task.csv"
        performance_by_task.to_csv(performance_by_task_path, index=False)
        print(f"Saved performance-by-task CSV: {performance_by_task_path}")
        print("\n=== Best Performance Per Task ===")
        print(performance_by_task.to_string(index=False, float_format=lambda x: _format_value(x)))


if __name__ == "__main__":
    main()
