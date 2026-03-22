from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from .common import best_reference_cer, best_reference_wer
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a Greek summarization system.
Produce a concise summary in Greek and return only the summary."""


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv, random_state
    dataset = load_dataset("DominusTea/GreekLegalSum", split="train")
    df = pd.DataFrame(dataset)
    df["subset"] = df["subset"].astype(int)
    subset_map = {"train": 0, "validation": 1, "test": 2}
    filtered = df.loc[df["subset"] == subset_map[split]].drop(columns=["subset"]).reset_index(drop=True)
    return filtered.rename(columns={"summary": "reference_summary"})


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "Given a Greek legal text, provide its summary also in Greek. "
        "Only the summary should be returned.\n\n"
        f"Text: {example['text']}\n"
        "Summary:"
    )


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "task": "summarization",
                "model": model,
                "samples": len(group),
                "wer_vs_reference": group.apply(
                    lambda row: best_reference_wer(row["prediction"], row["reference_summary"]),
                    axis=1,
                ).mean(),
                "cer_vs_reference": group.apply(
                    lambda row: best_reference_cer(row["prediction"], row["reference_summary"]),
                    axis=1,
                ).mean(),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["wer_vs_reference", "cer_vs_reference"], ascending=[True, True]).reset_index(drop=True)


def build_task() -> TaskSpec:
    return TaskSpec(
        name="summarization",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        evaluate=_evaluate,
    )
