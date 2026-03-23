from __future__ import annotations

import pandas as pd

from .common import evaluate_sequence_labeling, normalize_whitespace, parse_label_sequence
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek named entity recognition system.
Return only a Python-style list of tags and nothing else."""

NER_LABELS = [
    "S-LOC",
    "B-LOC",
    "I-LOC",
    "E-LOC",
    "S-ORG",
    "B-ORG",
    "I-ORG",
    "E-ORG",
    "S-PERSON",
    "B-PERSON",
    "I-PERSON",
    "E-PERSON",
    "S-MISC",
    "B-MISC",
    "I-MISC",
    "E-MISC",
    "O",
]


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del random_state
    import data_wrapper

    datasets = pd.read_csv(data_csv)
    df = data_wrapper.BarziokasDt(datasets=datasets).get(split).copy()
    df["sentence"] = df["sentence"].apply(list)
    df["labels"] = df["ne_tag4"].apply(lambda tags: [normalize_whitespace(tag) for tag in list(tags)])
    return df[["sentence", "labels"]].reset_index(drop=True)


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "You are given a tokenized Greek sentence. Return only a Python-style list of NER tags, one tag per token.\n"
        f"Use only these labels: {NER_LABELS}\n\n"
        f"Tokens: {example['sentence']}\n"
        "Tags:"
    )


def _normalize_prediction(text: str, _example: dict[str, object]) -> list[str]:
    return parse_label_sequence(text, NER_LABELS)


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    summary = evaluate_sequence_labeling(raw, task_name="ner", gold_col="labels", labels=NER_LABELS, filler_label="O")

    # Added metric: entity-level F1 (strict IOBES via seqeval).
    entity_f1_by_model = {}
    for model, group in raw.groupby("model", sort=False):
        try:
            from seqeval.metrics import f1_score as seqeval_f1_score
            from seqeval.scheme import IOBES
        except Exception:
            entity_f1_by_model[model] = float("nan")
            continue

        gold_sequences = []
        pred_sequences = []
        for row in group.itertuples(index=False):
            gold_tags = [normalize_whitespace(tag) for tag in list(row.labels)]
            pred_tags = list(row.prediction)[: len(gold_tags)]
            if len(pred_tags) < len(gold_tags):
                pred_tags.extend(["O"] * (len(gold_tags) - len(pred_tags)))
            gold_sequences.append(gold_tags)
            pred_sequences.append(pred_tags)

        entity_f1_by_model[model] = float(
            seqeval_f1_score(gold_sequences, pred_sequences, mode="strict", scheme=IOBES)
        )

    summary["entity_f1"] = summary["model"].map(entity_f1_by_model)
    return summary


def build_task() -> TaskSpec:
    return TaskSpec(
        name="ner",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
