from __future__ import annotations

import argparse

from benchmark_suite import GenerationConfig, list_tasks, run_task, save_run_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark tasks against Ollama models.")
    parser.add_argument("--task", required=True, choices=list_tasks(), help="Benchmark task to run.")
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model names to benchmark.")
    parser.add_argument("--sample-size", type=int, default=100, help="How many examples to score.")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--data-csv", default="data.csv", help="Path to the benchmark dataset registry CSV.")
    parser.add_argument("--output-dir", default="results/suite", help="Where to save benchmark outputs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to Ollama.")
    parser.add_argument("--num-predict", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-request timeout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, raw = run_task(
        task_name=args.task,
        models=args.models,
        sample_size=args.sample_size,
        random_state=args.random_state,
        data_csv=args.data_csv,
        config=GenerationConfig(
            temperature=args.temperature,
            num_predict=args.num_predict,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    save_run_outputs(summary, raw, args.output_dir, args.task)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
