from __future__ import annotations

import pandas as pd

from .common import evaluate_sequence_labeling, normalize_whitespace, parse_label_sequence
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek POS tagging system.
Return only a Python-style list of tags and nothing else."""


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del random_state
    import data_wrapper

    datasets = pd.read_csv(data_csv)
    token_df = data_wrapper.ProkopidisUdDt(datasets=datasets).get(split).copy()
    label_space = sorted(token_df["x"].dropna().astype(str).map(normalize_whitespace).unique())

    grouped = pd.DataFrame()
    grouped["tokens"] = token_df.groupby("s").w.apply(list)
    grouped["labels"] = token_df.groupby("s").x.apply(lambda values: [normalize_whitespace(value) for value in list(values)])
    grouped = grouped.reset_index(drop=True)
    grouped["label_space"] = ", ".join(label_space)
    return grouped


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "You are given a tokenized Greek sentence. Return only a Python-style list of POS tags, one tag per token.\n"
        f"Use only these labels: {example['label_space']}\n\n"
        f"Tokens: {example['tokens']}\n"
        "Tags:"
    )


def _normalize_prediction(text: str, example: dict[str, object]) -> list[str]:
    labels = [normalize_whitespace(label) for label in str(example["label_space"]).split(",")]
    return parse_label_sequence(text, labels)


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    label_space = sorted({tag for row in raw["labels"] for tag in row})
    filler = "_" if "_" in label_space else label_space[0]
    return evaluate_sequence_labeling(raw, task_name="pos", gold_col="labels", labels=label_space, filler_label=filler)


def build_task() -> TaskSpec:
    return TaskSpec(
        name="pos",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
