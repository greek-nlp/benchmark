from __future__ import annotations

from functools import reduce

import pandas as pd
from datasets import load_dataset

from .common import evaluate_classification, normalize_whitespace
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek legal text classification system.
Return only one label from the provided label set and nothing else."""


def _load_subset(subset_name: str, split: str) -> pd.DataFrame:
    dataset = load_dataset("AI-team-UoA/greek_legal_code", subset_name)
    resolved_split = split if split in dataset else next(iter(dataset.keys()))
    subset_df = pd.DataFrame(dataset[resolved_split]).rename(columns={"label": subset_name})
    keep_columns = [column for column in ["text", subset_name] if column in subset_df.columns]
    return subset_df[keep_columns].reset_index(drop=True)


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv, random_state
    frames = [_load_subset(subset_name, split) for subset_name in ["volume", "chapter", "subject"]]
    merged = reduce(lambda left, right: left.merge(right, left_index=True, right_index=True, how="inner"), frames)

    text_columns = [column for column in merged.columns if column.startswith("text")]
    if "text" not in merged.columns and text_columns:
        merged["text"] = merged[text_columns[0]]
    drop_columns = [column for column in text_columns if column != "text"]
    if drop_columns:
        merged = merged.drop(columns=drop_columns)

    merged = merged[["text", "volume", "chapter", "subject"]].dropna().reset_index(drop=True)
    merged["subject"] = merged["subject"].map(normalize_whitespace)
    merged["label_space"] = ", ".join(sorted(merged["subject"].unique()))
    return merged


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "Given the following Greek legal text, return only its legal subject label.\n"
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
    raw["subject"] = raw["subject"].map(normalize_whitespace)
    raw["prediction"] = raw["prediction"].map(normalize_whitespace)
    return evaluate_classification(raw, label_col="subject", task_name="legal_classification")


def build_task() -> TaskSpec:
    return TaskSpec(
        name="legal_classification",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
