from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from legal_label_metadata import get_legal_volume_name
from .common import evaluate_classification, normalize_whitespace
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek legal text classification system.
Return only one label from the provided label set and nothing else."""


TARGET_LABEL_COLUMN = "volume"
_LOCAL_PARQUET_SPLITS = {
    "train": Path("/tmp/glc_volume_train.parquet"),
    "validation": Path("/tmp/glc_volume_validation.parquet"),
    "test": Path("/tmp/glc_volume_test.parquet"),
}


def _extract_label_names(dataset, label_column: str = "label") -> list[str] | None:
    try:
        split_name = next(iter(dataset.keys()))
        feature = dataset[split_name].features.get(label_column)
    except Exception:
        return None

    names = getattr(feature, "names", None)
    if names:
        return list(names)

    int2str = getattr(feature, "int2str", None)
    num_classes = getattr(feature, "num_classes", None)
    if callable(int2str) and isinstance(num_classes, int):
        try:
            return [normalize_whitespace(int2str(index)) for index in range(num_classes)]
        except Exception:
            return None
    return None


def _fallback_label_names(subset_name: str) -> list[str] | None:
    if subset_name == "volume":
        return [name for name in (get_legal_volume_name(index) for index in range(47)) if name]
    return None


def _legal_label(value: object, *, name: object | None = None) -> str:
    normalized_name = normalize_whitespace(name) if name is not None and pd.notna(name) else ""
    if normalized_name:
        return normalized_name
    return f"Volume {normalize_whitespace(value)}"


def _volume_sort_key(label: str) -> tuple[int, str]:
    suffix = label.removeprefix("Volume ").strip()
    return (int(suffix), label) if suffix.isdigit() else (10**9, label)


def _load_subset(subset_name: str, split: str) -> pd.DataFrame:
    dataset = load_dataset("AI-team-UoA/greek_legal_code", subset_name)
    resolved_split = split if split in dataset else next(iter(dataset.keys()))
    subset_df = pd.DataFrame(dataset[resolved_split]).rename(columns={"label": subset_name})
    label_names = _extract_label_names(dataset) or _fallback_label_names(subset_name)
    if label_names and subset_name in subset_df.columns:
        subset_df[f"{subset_name}_name"] = subset_df[subset_name].map(
            lambda value: label_names[int(value)] if pd.notna(value) and int(value) < len(label_names) else pd.NA
        )
    keep_columns = [
        column
        for column in ["text", subset_name, f"{subset_name}_name"]
        if column in subset_df.columns
    ]
    return subset_df[keep_columns].reset_index(drop=True)


def _load_local_volume_split(split: str) -> pd.DataFrame | None:
    parquet_path = _LOCAL_PARQUET_SPLITS.get(split)
    if parquet_path is None or not parquet_path.exists():
        return None

    subset_df = pd.read_parquet(parquet_path).rename(columns={"label": "volume"})
    subset_df["volume_name"] = subset_df["volume"].map(lambda value: get_legal_volume_name(value) or pd.NA)
    keep_columns = [column for column in ["text", "volume", "volume_name"] if column in subset_df.columns]
    return subset_df[keep_columns].reset_index(drop=True)


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv, random_state
    try:
        merged = _load_subset("volume", split)
    except Exception:
        merged = _load_local_volume_split(split)
        if merged is None:
            raise

    merged = merged[["text", "volume", *([column for column in ["volume_name"] if column in merged.columns])]].dropna(
        subset=["text", "volume"]
    ).reset_index(drop=True)
    merged["chapter"] = pd.NA
    merged["subject"] = pd.NA
    merged["volume"] = merged["volume"].map(normalize_whitespace)
    optional_columns = [column for column in ["volume_name"] if column in merged.columns]
    for column in optional_columns:
        merged[column] = merged[column].map(normalize_whitespace)
    target_name_column = f"{TARGET_LABEL_COLUMN}_name"
    if target_name_column in merged.columns:
        merged["label"] = merged.apply(
            lambda row: _legal_label(row[TARGET_LABEL_COLUMN], name=row[target_name_column]),
            axis=1,
        )
    else:
        merged["label"] = merged[TARGET_LABEL_COLUMN].map(_legal_label)
    merged["label_space"] = ", ".join(sorted(merged["label"].unique(), key=_volume_sort_key))
    return merged


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "Given the following Greek legal text, return only its coarse-grained legal volume label.\n"
        f"Choose exactly one label from: {example['label_space']}\n\n"
        f"Text: {example['text']}\n"
        "Label:"
    )


def _normalize_prediction(text: str, example: dict[str, object]) -> str:
    prediction = normalize_whitespace(text)
    valid_labels = [normalize_whitespace(label) for label in str(example["label_space"]).split(",")]
    for label in valid_labels:
        if prediction.lower() == label.lower():
            return label
    lowered = prediction.lower()
    for label in valid_labels:
        if label.lower() in lowered:
            return label
    return prediction


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["label"] = raw["label"].map(normalize_whitespace)
    raw["prediction"] = raw["prediction"].map(normalize_whitespace)
    return evaluate_classification(raw, label_col="label", task_name="legal_classification")


def build_task() -> TaskSpec:
    return TaskSpec(
        name="legal_classification",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
