from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_METRIC_BY_TASK = {
    "gec": ("gleu_vs_reference", False),
    "intent_classification": ("macro_f1", False),
    "legal_classification": ("macro_f1", False),
    "machine_translation": ("chrf", False),
    "ner": ("macro_f1", False),
    "pos": ("macro_f1", False),
    "summarization": ("bertscore_f1", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a completed benchmark run into leaderboard-friendly JSON artifacts.")
    parser.add_argument(
        "--results-dir",
        default="results/full_benchmark_suite",
        help="Directory containing a completed benchmark run with *_summary.csv and *_predictions.csv files.",
    )
    parser.add_argument(
        "--benchmark-name",
        default="greek-nlp-benchmark",
        help="Benchmark identifier to write into exported JSON rows.",
    )
    parser.add_argument(
        "--benchmark-version",
        default="v1",
        help="Benchmark version tag to write into exported JSON rows.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt template version tag for leaderboard metadata.",
    )
    parser.add_argument(
        "--backend",
        default="ollama",
        help="Inference backend label for leaderboard metadata.",
    )
    return parser.parse_args()


def _run_id(results_dir: Path) -> str:
    return results_dir.name


def _load_summary_frames(results_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("*_summary.csv")):
        task_name = path.name.removesuffix("_summary.csv")
        if task_name == "all_tasks":
            continue
        frame = pd.read_csv(path)
        if "task" not in frame.columns:
            frame["task"] = task_name
        frames[task_name] = frame
    return frames


def _jsonify_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _row_metrics(row: pd.Series) -> dict[str, object]:
    excluded = {
        "task",
        "model",
        "target_lang",
        "samples",
        "avg_latency_seconds",
        "quality_score",
        "segment_label",
        "primary_metric",
        "lower_is_better",
    }
    metrics: dict[str, object] = {}
    for column, value in row.items():
        if column in excluded:
            continue
        if pd.isna(value):
            continue
        metrics[column] = _jsonify_scalar(value)
    return metrics


def _segment_key(task_name: str, row: pd.Series) -> str:
    if task_name == "machine_translation" and "target_lang" in row and pd.notna(row["target_lang"]):
        return f"{task_name}_{str(row['target_lang']).lower()}"
    return task_name


def _primary_metrics(summary_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for task_name, frame in summary_frames.items():
        metric = PRIMARY_METRIC_BY_TASK[task_name][0]
        if task_name == "summarization" and (metric not in frame.columns or frame[metric].isna().all()):
            metric = "rouge_l"
        chosen[task_name] = metric
    return chosen


def _build_combined(summary_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for task_name, frame in summary_frames.items():
        metric, lower_is_better = PRIMARY_METRIC_BY_TASK[task_name]
        if task_name == "summarization" and (metric not in frame.columns or frame[metric].isna().all()):
            metric = "rouge_l"

        working = frame.copy()
        if "task" not in working.columns:
            working["task"] = task_name

        if task_name == "machine_translation":
            for target_lang, target_df in working.groupby("target_lang", sort=False):
                segment = target_df.copy()
                segment["segment_label"] = f"{task_name}_{str(target_lang).lower()}"
                segment["primary_metric"] = metric
                segment["lower_is_better"] = lower_is_better
                rows.append(segment)
        else:
            working["segment_label"] = task_name
            working["primary_metric"] = metric
            working["lower_is_better"] = lower_is_better
            rows.append(working)

    combined = pd.concat(rows, ignore_index=True)
    combined["quality_score"] = np.nan

    for segment_label, segment_df in combined.groupby("segment_label", sort=False):
        metric = str(segment_df["primary_metric"].iloc[0])
        lower_is_better = bool(segment_df["lower_is_better"].iloc[0])
        values = segment_df[metric].astype(float)
        if values.nunique(dropna=True) <= 1:
            combined.loc[segment_df.index, "quality_score"] = 50.0
            continue
        if lower_is_better:
            scaled = 100.0 * (values.max() - values) / (values.max() - values.min())
        else:
            scaled = 100.0 * (values - values.min()) / (values.max() - values.min())
        combined.loc[segment_df.index, "quality_score"] = scaled

    return combined


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
    grouped["rank"] = range(1, len(grouped) + 1)
    return grouped


def _export_results_jsonl(
    *,
    results_dir: Path,
    summary_frames: dict[str, pd.DataFrame],
    benchmark_name: str,
    benchmark_version: str,
    prompt_version: str,
    backend: str,
) -> Path:
    combined = _build_combined(summary_frames)
    overall = _overall_model_table(combined).set_index("model")
    primary_metrics = _primary_metrics(summary_frames)

    rows: list[dict[str, object]] = []
    for model in combined["model"].drop_duplicates().tolist():
        model_rows = combined.loc[combined["model"] == model].copy()
        metrics: dict[str, dict[str, object]] = {}
        samples: dict[str, int] = {}
        primary_by_segment: dict[str, dict[str, object]] = {}

        for row in model_rows.to_dict(orient="records"):
            series = pd.Series(row)
            task_name = str(series["task"])
            segment = _segment_key(task_name, series)
            metrics[segment] = _row_metrics(series)
            if "samples" in series and pd.notna(series["samples"]):
                samples[segment] = int(series["samples"])
            primary_metric = primary_metrics[task_name]
            if primary_metric in series and pd.notna(series[primary_metric]):
                primary_by_segment[segment] = {
                    "metric": primary_metric,
                    "value": _jsonify_scalar(series[primary_metric]),
                }

        aggregate = overall.loc[model]
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "benchmark_version": benchmark_version,
                "run_id": _run_id(results_dir),
                "model_name": model,
                "backend": backend,
                "metrics": metrics,
                "primary_metrics": primary_by_segment,
                "samples": samples,
                "aggregate": {
                    "rank": int(aggregate["rank"]),
                    "tasks_completed": int(aggregate["tasks_completed"]),
                    "avg_normalized_quality": float(aggregate["avg_normalized_quality"]),
                    "median_normalized_quality": float(aggregate["median_normalized_quality"]),
                    "avg_latency_seconds": float(aggregate["avg_latency_seconds"]),
                },
                "metadata": {
                    "prompt_version": prompt_version,
                },
            }
        )

    output_path = results_dir / "leaderboard_results.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def _export_submission_files(
    *,
    results_dir: Path,
    benchmark_name: str,
    benchmark_version: str,
    prompt_version: str,
    backend: str,
) -> Path:
    predictions_path = results_dir / "all_tasks_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing combined predictions file: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    if "example_id" not in predictions.columns:
        predictions = predictions.copy()
        group_columns = ["task", "model"]
        if "target_lang" in predictions.columns:
            target_lang_series = predictions["target_lang"].fillna("")
            if target_lang_series.ne("").any():
                group_columns.append("target_lang")

        positions = predictions.groupby(group_columns, sort=False, dropna=False).cumcount()
        if "target_lang" in group_columns:
            target_lang_values = predictions["target_lang"].fillna("unknown").astype(str)
            predictions["example_id"] = [
                (
                    f"{task}:{target_lang}:{int(position):06d}"
                    if task == "machine_translation" and target_lang != "unknown"
                    else f"{task}:{int(position):06d}"
                )
                for task, target_lang, position in zip(
                    predictions["task"].astype(str),
                    target_lang_values,
                    positions,
                )
            ]
        else:
            predictions["example_id"] = [
                f"{task}:{int(position):06d}"
                for task, position in zip(predictions["task"].astype(str), positions)
            ]

    submissions_dir = results_dir / "leaderboard_submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    for model, model_df in predictions.groupby("model", sort=False):
        entries: list[dict[str, object]] = []
        for row in model_df.to_dict(orient="records"):
            task_name = str(row["task"])
            entry = {
                "task": task_name,
                "example_id": row["example_id"],
                "prediction": _jsonify_scalar(row.get("prediction")),
            }
            if pd.notna(row.get("target_lang")):
                entry["target_lang"] = str(row["target_lang"])
            entries.append(entry)

        payload = {
            "benchmark_name": benchmark_name,
            "benchmark_version": benchmark_version,
            "run_id": _run_id(results_dir),
            "model_name": model,
            "backend": backend,
            "metadata": {
                "prompt_version": prompt_version,
            },
            "predictions": entries,
        }

        safe_name = model.replace("/", "__").replace(":", "__")
        output_path = submissions_dir / f"{safe_name}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return submissions_dir


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_frames = _load_summary_frames(results_dir)
    if not summary_frames:
        raise FileNotFoundError(f"No per-task summary CSVs found under {results_dir}")

    results_path = _export_results_jsonl(
        results_dir=results_dir,
        summary_frames=summary_frames,
        benchmark_name=args.benchmark_name,
        benchmark_version=args.benchmark_version,
        prompt_version=args.prompt_version,
        backend=args.backend,
    )
    submissions_dir = _export_submission_files(
        results_dir=results_dir,
        benchmark_name=args.benchmark_name,
        benchmark_version=args.benchmark_version,
        prompt_version=args.prompt_version,
        backend=args.backend,
    )

    print(f"Saved leaderboard results JSONL: {results_path}")
    print(f"Saved leaderboard submission files: {submissions_dir}")


if __name__ == "__main__":
    main()
