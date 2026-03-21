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


def parse_args() -> argparse.Namespace:
    task_choices = [ALL_TASKS, *list_tasks()]
    parser = argparse.ArgumentParser(description="Run benchmark tasks against Ollama models.")
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
    parser.add_argument("--sample-size", type=int, default=100, help="How many examples to score.")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--data-csv", default="data.csv", help="Path to the benchmark dataset registry CSV.")
    parser.add_argument("--output-dir", default="results/suite", help="Where to save benchmark outputs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to Ollama.")
    parser.add_argument("--num-predict", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-request timeout.")
    return parser.parse_args()


def _selected_tasks(task_name: str) -> list[str]:
    if task_name == ALL_TASKS:
        return list_tasks()
    return [task_name]


def _print_summary(summary: pd.DataFrame) -> None:
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not math.isnan(x) else "nan"))


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    config = GenerationConfig(
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout_seconds=args.timeout_seconds,
    )

    combined_summaries: list[pd.DataFrame] = []
    selected_tasks = _selected_tasks(args.task)

    for task_name in selected_tasks:
        print(f"\n=== Running task: {task_name} ===")
        summary, raw = run_task(
            task_name=task_name,
            models=args.models,
            sample_size=args.sample_size,
            random_state=args.random_state,
            data_csv=args.data_csv,
            config=config,
        )
        save_run_outputs(summary, raw, output_path, task_name)
        combined_summaries.append(summary)
        _print_summary(summary)
        print(f"\nSaved summary CSV: {output_path / f'{task_name}_summary.csv'}")
        print(f"Saved predictions CSV: {output_path / f'{task_name}_predictions.csv'}")
        print(f"Saved visualization: {output_path / f'{task_name}_visualization.html'}")

    if len(combined_summaries) > 1:
        combined_summary = pd.concat(combined_summaries, ignore_index=True)
        combined_summary_path = output_path / "all_tasks_summary.csv"
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"\nSaved combined summary CSV: {combined_summary_path}")


if __name__ == "__main__":
    main()
