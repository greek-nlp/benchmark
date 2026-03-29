from __future__ import annotations

import pandas as pd

from .common import best_reference_cer, best_reference_wer
from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a translation system.
Translate the user's text accurately.
Return only the translation and nothing else."""

TARGET_LANGS = ["eng", "jpn", "fas"]


def _first_reference(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _load_dataset(
    *,
    data_csv: str,
    random_state: int,
    split: str = "test",
    target_lang_limits: dict[str, int] | None = None,
    full_corpus_target_langs: list[str] | None = None,
):
    import data_wrapper

    datasets = pd.read_csv(data_csv)
    mt_dataset = data_wrapper.ProkopidisMtDt(datasets=datasets)
    full_corpus_target_langs = set(full_corpus_target_langs or [])

    records = []
    for target_lang in TARGET_LANGS:
        if target_lang in full_corpus_target_langs:
            train_df = mt_dataset.get(target_lang=target_lang, split="train").copy()
            test_df = mt_dataset.get(target_lang=target_lang, split="test").copy()
            selected = pd.concat([train_df, test_df], ignore_index=True)
        else:
            selected = mt_dataset.get(target_lang=target_lang, split=split).copy()
        selected["target_lang"] = target_lang
        selected = selected.reset_index(drop=True)
        if target_lang_limits and target_lang in target_lang_limits:
            selected = selected.head(target_lang_limits[target_lang]).reset_index(drop=True)
        records.append(selected)

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
        bleu_score = float("nan")
        chrf_score = float("nan")
        try:
            import sacrebleu

            predictions = group["prediction"].fillna("").astype(str).tolist()
            references = group["target"].apply(_first_reference).fillna("").astype(str).tolist()
            bleu_score = float(sacrebleu.corpus_bleu(predictions, [references]).score)
            chrf_score = float(sacrebleu.corpus_chrf(predictions, [references]).score)
        except Exception:
            pass

        rows.append(
            {
                "task": "machine_translation",
                "model": model,
                "target_lang": target_lang,
                "samples": len(group),
                "wer_vs_reference": group.apply(lambda row: best_reference_wer(row["prediction"], row["target"]), axis=1).mean(),
                "cer_vs_reference": group.apply(lambda row: best_reference_cer(row["prediction"], row["target"]), axis=1).mean(),
                "bleu": bleu_score,
                "chrf": chrf_score,
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
