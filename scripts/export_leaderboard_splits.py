from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.core import _ensure_example_ids
from benchmark_suite.registry import TASKS, list_tasks


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
        "machine_translation": {
            "target_lang_limits": {"eng": 500, "fas": 500, "jpn": 500},
            "full_corpus_target_langs": ["fas", "jpn"],
        },
    }
}


def _task_data_limit(task_name: str, cap_profile: str) -> int | None:
    return FULL_TEST_TASK_CAPS.get(cap_profile, {}).get(task_name)


def _task_specific_options(task_name: str, cap_profile: str) -> dict[str, object]:
    return dict(FULL_TEST_TASK_OPTIONS.get(cap_profile, {}).get(task_name, {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen leaderboard splits from the current benchmark loaders.")
    parser.add_argument(
        "--output-dir",
        default="leaderboard_data/v1",
        help="Directory where the frozen split files and manifest will be written.",
    )
    parser.add_argument(
        "--data-csv",
        default="data.csv",
        help="Path to the benchmark dataset registry CSV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed forwarded to task loaders that construct deterministic splits.",
    )
    parser.add_argument(
        "--task-cap-profile",
        choices=[NO_CAP_PROFILE, REASONABLE_CAP_PROFILE],
        default=REASONABLE_CAP_PROFILE,
        help="Deterministic cap profile to freeze into the exported leaderboard splits.",
    )
    parser.add_argument(
        "--benchmark-version",
        default="v1",
        help="Version label to store in the manifest.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list_tasks(),
        default=None,
        help="Optional subset of tasks to export.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl", "csv"],
        default="parquet",
        help="Dataset file format for exported splits.",
    )
    return parser.parse_args()


def _normalize_for_serialization(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: value if not isinstance(value, tuple) else list(value)
        )
    return normalized


def _file_name(task_name: str, frame: pd.DataFrame, file_format: str) -> str:
    if task_name == "machine_translation" and "target_lang" in frame.columns:
        target_lang = str(frame["target_lang"].iloc[0]).lower()
        return f"{task_name}_{target_lang}_test.{file_format}"
    return f"{task_name}_test.{file_format}"


def _write_frame(frame: pd.DataFrame, path: Path, *, file_format: str) -> None:
    if file_format == "parquet":
        frame.to_parquet(path, index=False)
    elif file_format == "jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_csv(path, index=False)


def _export_task_split(
    *,
    task_name: str,
    data_csv: str,
    random_state: int,
    cap_profile: str,
    output_dir: Path,
    file_format: str,
) -> list[dict[str, object]]:
    task = TASKS[task_name]
    task_options = _task_specific_options(task_name, cap_profile)
    data_limit = _task_data_limit(task_name, cap_profile)

    dataset = task.load_dataset(
        data_csv=data_csv,
        random_state=random_state,
        **task_options,
    )
    dataset = _ensure_example_ids(dataset, task_name=task_name)
    if data_limit is not None:
        dataset = dataset.head(data_limit).reset_index(drop=True)

    dataset = _normalize_for_serialization(dataset)
    manifests: list[dict[str, object]] = []

    if task_name == "machine_translation" and "target_lang" in dataset.columns:
        for target_lang, target_df in dataset.groupby("target_lang", sort=False):
            output_path = output_dir / _file_name(task_name, target_df, file_format)
            _write_frame(target_df.reset_index(drop=True), output_path, file_format=file_format)
            manifests.append(
                {
                    "task": task_name,
                    "segment": f"{task_name}_{str(target_lang).lower()}",
                    "split": "test",
                    "target_lang": str(target_lang),
                    "path": str(output_path.relative_to(output_dir)),
                    "rows": int(len(target_df)),
                    "columns": list(target_df.columns),
                }
            )
        return manifests

    output_path = output_dir / _file_name(task_name, dataset, file_format)
    _write_frame(dataset.reset_index(drop=True), output_path, file_format=file_format)
    manifests.append(
        {
            "task": task_name,
            "segment": task_name,
            "split": "test",
            "path": str(output_path.relative_to(output_dir)),
            "rows": int(len(dataset)),
            "columns": list(dataset.columns),
        }
    )
    return manifests


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_entries: list[dict[str, object]] = []
    selected_tasks = args.tasks or list_tasks()
    for task_name in selected_tasks:
        split_entries.extend(
            _export_task_split(
                task_name=task_name,
                data_csv=args.data_csv,
                random_state=args.random_state,
                cap_profile=args.task_cap_profile,
                output_dir=output_dir,
                file_format=args.format,
            )
        )

    manifest = {
        "benchmark_version": args.benchmark_version,
        "task_cap_profile": args.task_cap_profile,
        "random_state": args.random_state,
        "file_format": args.format,
        "tasks": selected_tasks,
        "splits": split_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved leaderboard splits under: {output_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
