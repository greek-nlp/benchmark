from __future__ import annotations

import io
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import download_bytes, evaluate_classification, normalize_whitespace
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek intent classification system.
Return only the intent label and nothing else."""


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv
    zip_bytes = download_bytes("https://msensis.com/wp-content/uploads/2023/06/uniway.zip")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        files = {
            "text": "uniway/GR/corpus.txt",
            "ne_tags": "uniway/GR/entities.txt",
            "label": "uniway/GR/intents.txt",
        }
        columns = {}
        for column, inner_path in files.items():
            with archive.open(inner_path) as handle:
                columns[column] = [line.decode("utf-8").strip() for line in handle.readlines()]

    df = pd.DataFrame(columns)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        shuffle=True,
        random_state=random_state,
    )
    selected = train_df if split == "train" else test_df
    selected = selected.reset_index(drop=True)
    selected["label_space"] = ", ".join(sorted(selected["label"].unique()))
    return selected


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "Given a text, provide the intent of the text. Only the intent should be returned.\n"
        f"Here is the list of possible intents: {example['label_space']}\n\n"
        f"Text: {example['text']}\n"
        "Intent:"
    )


def _normalize_prediction(text: str, example: dict[str, object]) -> str:
    prediction = normalize_whitespace(text)
    valid_labels = [label.strip() for label in str(example["label_space"]).split(",")]
    for label in valid_labels:
        if prediction.lower() == label.lower():
            return label
    return prediction


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    return evaluate_classification(raw, label_col="label", task_name="intent_classification")


def build_task() -> TaskSpec:
    return TaskSpec(
        name="intent_classification",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
