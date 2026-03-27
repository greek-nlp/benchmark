from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite import GenerationConfig, list_tasks, run_task, save_run_outputs

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct",
    "llama3.1:8b-instruct",
    "aya-expanse:8b",
    "gemma2:9b",
    "falcon3:7b-instruct",
]

ALL_TASKS = "all"
NO_CAP_PROFILE = "none"
REASONABLE_CAP_PROFILE = "reasonable"

FULL_TEST_TASK_CAPS = {
    REASONABLE_CAP_PROFILE: {
        "legal_classification": 500,
        "ner": 500,
        "summarization": 300,
    }
}

FULL_TEST_TASK_OPTIONS = {
    REASONABLE_CAP_PROFILE: {
        "machine_translation": {"target_lang_limits": {"eng": 300, "fas": 4, "jpn": 9}},
    }
}


def parse_args() -> argparse.Namespace:
    task_choices = [ALL_TASKS, *list_tasks()]
    parser = argparse.ArgumentParser(description="Run the full Greek NLP benchmark suite against Ollama models.")
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
        help="How many examples to score. Use 0 to run on the full test dataset.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated sampled runs (Monte Carlo style). Use >1 with --sample-size > 0.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--data-csv", default="data.csv", help="Path to the benchmark dataset registry CSV.")
    parser.add_argument("--output-dir", default="results/full_benchmark_suite", help="Where to save benchmark outputs.")
    parser.add_argument(
        "--task-cap-profile",
        choices=[NO_CAP_PROFILE, REASONABLE_CAP_PROFILE],
        default=NO_CAP_PROFILE,
        help="Optional deterministic per-task cap profile. Applies first-instance caps before any sampling.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to Ollama.")
    parser.add_argument("--num-predict", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-request timeout.")
    return parser.parse_args()


def _selected_tasks(task_name: str) -> list[str]:
    if task_name == ALL_TASKS:
        return list_tasks()
    return [task_name]


def _task_data_limit(task_name: str, cap_profile: str) -> int | None:
    return FULL_TEST_TASK_CAPS.get(cap_profile, {}).get(task_name)


def _task_specific_options(task_name: str, cap_profile: str) -> dict[str, object]:
    return dict(FULL_TEST_TASK_OPTIONS.get(cap_profile, {}).get(task_name, {}))


def _print_summary(summary: pd.DataFrame) -> None:
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not math.isnan(x) else "nan"))


def _aggregate_repeated_summaries(summary_by_repeat: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated_summary = pd.concat(summary_by_repeat, ignore_index=True)
    key_columns = [column for column in ["task", "model", "target_lang"] if column in repeated_summary.columns]
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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    sample_size = None if args.sample_size <= 0 else args.sample_size
    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1.")
    if args.repeats > 1 and sample_size is None:
        raise ValueError("Monte Carlo mode requires a specific sample size. Set --sample-size > 0.")

    config = GenerationConfig(
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout_seconds=args.timeout_seconds,
    )

    combined_summaries: list[pd.DataFrame] = []
    combined_predictions: list[pd.DataFrame] = []
    selected_tasks = _selected_tasks(args.task)

    print("Tasks:", selected_tasks)
    print("Models:", args.models)
    print("Repeats:", args.repeats)

    for task_name in selected_tasks:
        print(f"\n=== Running task: {task_name} ===")
        task_data_limit = _task_data_limit(task_name, args.task_cap_profile)
        task_options = _task_specific_options(task_name, args.task_cap_profile)
        if args.repeats == 1:
            summary, raw = run_task(
                task_name=task_name,
                models=args.models,
                sample_size=sample_size,
                data_limit=task_data_limit,
                random_state=args.random_state,
                data_csv=args.data_csv,
                config=config,
                task_options=task_options,
            )
            save_run_outputs(summary, raw, output_path, task_name)
            combined_summaries.append(summary.assign(task_name=task_name))
            combined_predictions.append(raw.assign(task_name=task_name))
            _print_summary(summary)
            print(f"\nSaved summary CSV: {output_path / f'{task_name}_summary.csv'}")
            print(f"Saved predictions CSV: {output_path / f'{task_name}_predictions.csv'}")
            print(f"Saved visualization: {output_path / f'{task_name}_visualization.html'}")
        else:
            repeat_summaries: list[pd.DataFrame] = []
            repeat_predictions: list[pd.DataFrame] = []
            for repeat_index in range(args.repeats):
                repeat_number = repeat_index + 1
                repeat_seed = args.random_state + repeat_index
                repeat_output_dir = output_path / task_name / f"repeat_{repeat_number:02d}"
                print(f"Running repeat {repeat_number}/{args.repeats} with seed {repeat_seed}")
                summary, raw = run_task(
                    task_name=task_name,
                    models=args.models,
                    sample_size=sample_size,
                    data_limit=task_data_limit,
                    random_state=repeat_seed,
                    data_csv=args.data_csv,
                    config=config,
                    task_options=task_options,
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

            task_output_dir = output_path / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            aggregated_summary_path = task_output_dir / f"{task_name}_summary_with_sem.csv"
            repeated_summary_path = task_output_dir / f"{task_name}_repeat_summaries.csv"
            repeated_predictions_path = task_output_dir / f"{task_name}_repeat_predictions.csv"

            aggregated_summary.to_csv(aggregated_summary_path, index=False)
            repeated_summary.to_csv(repeated_summary_path, index=False)
            repeated_predictions.to_csv(repeated_predictions_path, index=False)
            combined_summaries.append(aggregated_summary.assign(task_name=task_name))
            combined_predictions.append(repeated_predictions.assign(task_name=task_name))
            _print_summary(aggregated_summary)
            print(f"\nSaved aggregated summary CSV: {aggregated_summary_path}")
            print(f"Saved repeat summaries CSV: {repeated_summary_path}")
            print(f"Saved repeat predictions CSV: {repeated_predictions_path}")

    if len(combined_summaries) > 1:
        combined_summary = pd.concat(combined_summaries, ignore_index=True)
        combined_summary_path = output_path / ("all_tasks_summary_with_sem.csv" if args.repeats > 1 else "all_tasks_summary.csv")
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"\nSaved combined summary CSV: {combined_summary_path}")
    if len(combined_predictions) > 1:
        combined_predictions_df = pd.concat(combined_predictions, ignore_index=True)
        combined_predictions_path = output_path / (
            "all_tasks_repeat_predictions.csv" if args.repeats > 1 else "all_tasks_predictions.csv"
        )
        combined_predictions_df.to_csv(combined_predictions_path, index=False)
        print(f"Saved combined predictions CSV: {combined_predictions_path}")


if __name__ == "__main__":
    main()
