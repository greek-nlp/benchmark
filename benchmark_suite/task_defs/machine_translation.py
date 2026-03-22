from __future__ import annotations

from sklearn.model_selection import train_test_split

from .common import best_reference_cer, best_reference_wer, download_bytes, load_dataset_registry, parse_tmx_records
import pandas as pd

from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a translation system.
Translate the user's text accurately.
Return only the translation and nothing else."""

TARGET_LANGS = ["eng", "jpn", "fas"]


def _load_dataset(*, data_csv: str, random_state: int, split: str = "test"):
    registry = load_dataset_registry(data_csv)
    repo_url = registry.loc[registry["id"] == 486, "url"].iloc[0]

    records = []
    for target_lang in TARGET_LANGS:
        zip_bytes = download_bytes(f"{repo_url}archives/ell-{target_lang}.zip")
        lang_df = parse_tmx_records(zip_bytes, target_langs=[target_lang]).drop_duplicates(subset=["source", "target"])
        train_df, test_df = train_test_split(
            lang_df,
            test_size=0.2,
            shuffle=True,
            random_state=random_state,
        )
        selected = train_df if split == "train" else test_df
        records.append(selected.reset_index(drop=True))

    return pd.DataFrame.from_records(
        [row for frame in records for row in frame.to_dict(orient="records")]
    ).reset_index(drop=True)


def _build_prompt(example: dict[str, object]) -> str:
    return (
        f"Given a text in Greek, translate it to {example['target_lang']}.\n"
        "Only the translation should be returned.\n\n"
        f"Text: {example['source']}\n"
        "Translation:"
    )


def _evaluate(raw):
    rows = []
    for (model, target_lang), group in raw.groupby(["model", "target_lang"], sort=False):
        rows.append(
            {
                "task": "machine_translation",
                "model": model,
                "target_lang": target_lang,
                "samples": len(group),
                "wer_vs_reference": group.apply(lambda row: best_reference_wer(row["prediction"], row["target"]), axis=1).mean(),
                "cer_vs_reference": group.apply(lambda row: best_reference_cer(row["prediction"], row["target"]), axis=1).mean(),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["target_lang", "wer_vs_reference", "cer_vs_reference"]).reset_index(drop=True)


def build_task() -> TaskSpec:
    return TaskSpec(
        name="machine_translation",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        evaluate=_evaluate,
    )
