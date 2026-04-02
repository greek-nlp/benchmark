from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a leaderboard submission JSON against a frozen benchmark manifest.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to leaderboard_data/<version>/manifest.json.",
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to a model submission JSON file exported by scripts/export_leaderboard_results.py.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_split(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported split file format: {path}")


def _expected_segments(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    splits = manifest.get("splits", [])
    if not isinstance(splits, list):
        raise ValueError("Manifest 'splits' must be a list.")
    segments: dict[str, dict[str, object]] = {}
    for entry in splits:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest split entry must be an object.")
        segment = entry.get("segment")
        if not isinstance(segment, str):
            raise ValueError("Each manifest split entry must include a string 'segment'.")
        segments[segment] = entry
    return segments


def _task_to_segment(task: str, target_lang: object) -> str:
    if task == "machine_translation":
        if not isinstance(target_lang, str) or not target_lang.strip():
            raise ValueError("Machine translation submissions must include a non-empty 'target_lang'.")
        return f"{task}_{target_lang.lower()}"
    return task


def _validate_submission_structure(submission: dict[str, object]) -> list[dict[str, object]]:
    required_top_level = [
        "benchmark_name",
        "benchmark_version",
        "run_id",
        "model_name",
        "backend",
        "predictions",
    ]
    missing = [field for field in required_top_level if field not in submission]
    if missing:
        raise ValueError(f"Submission is missing required top-level fields: {', '.join(missing)}")

    predictions = submission["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("Submission field 'predictions' must be a list.")

    normalized: list[dict[str, object]] = []
    for index, entry in enumerate(predictions):
        if not isinstance(entry, dict):
            raise ValueError(f"Prediction #{index} is not an object.")
        for field in ["task", "example_id", "prediction"]:
            if field not in entry:
                raise ValueError(f"Prediction #{index} is missing required field '{field}'.")
        normalized.append(entry)
    return normalized


def _validate_against_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    submission: dict[str, object],
    predictions: list[dict[str, object]],
) -> list[str]:
    benchmark_version = manifest.get("benchmark_version")
    if submission.get("benchmark_version") != benchmark_version:
        raise ValueError(
            f"Submission benchmark version '{submission.get('benchmark_version')}' does not match manifest version '{benchmark_version}'."
        )

    segments = _expected_segments(manifest)
    manifest_root = manifest_path.parent

    expected_ids_by_segment: dict[str, set[str]] = {}
    for segment, entry in segments.items():
        split_path = manifest_root / str(entry["path"])
        frame = _load_split(split_path)
        if "example_id" not in frame.columns:
            raise ValueError(f"Split file {split_path} is missing 'example_id'.")
        expected_ids_by_segment[segment] = set(frame["example_id"].astype(str))

    seen_by_segment: dict[str, set[str]] = {segment: set() for segment in segments}
    warnings: list[str] = []

    for index, entry in enumerate(predictions):
        task = str(entry["task"])
        target_lang = entry.get("target_lang")
        segment = _task_to_segment(task, target_lang)
        if segment not in expected_ids_by_segment:
            raise ValueError(f"Prediction #{index} targets unknown segment '{segment}'.")

        example_id = str(entry["example_id"])
        if example_id not in expected_ids_by_segment[segment]:
            raise ValueError(f"Prediction #{index} has unknown example_id '{example_id}' for segment '{segment}'.")
        if example_id in seen_by_segment[segment]:
            raise ValueError(f"Prediction #{index} duplicates example_id '{example_id}' for segment '{segment}'.")

        seen_by_segment[segment].add(example_id)

    missing_segments: list[str] = []
    for segment, expected_ids in expected_ids_by_segment.items():
        seen_ids = seen_by_segment[segment]
        if seen_ids != expected_ids:
            missing = len(expected_ids - seen_ids)
            extra = len(seen_ids - expected_ids)
            if missing or extra:
                missing_segments.append(f"{segment} (missing={missing}, extra={extra})")

    if missing_segments:
        raise ValueError("Submission does not cover the frozen split exactly: " + "; ".join(missing_segments))

    task_counts = {segment: len(ids) for segment, ids in expected_ids_by_segment.items()}
    warnings.append(
        "Validated segments: "
        + ", ".join(f"{segment}={task_counts[segment]}" for segment in sorted(task_counts))
    )
    return warnings


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    submission_path = Path(args.submission)

    manifest = _load_json(manifest_path)
    submission = _load_json(submission_path)
    predictions = _validate_submission_structure(submission)
    warnings = _validate_against_manifest(
        manifest_path=manifest_path,
        manifest=manifest,
        submission=submission,
        predictions=predictions,
    )

    print(f"Submission is valid: {submission_path}")
    for warning in warnings:
        print(warning)


if __name__ == "__main__":
    main()
