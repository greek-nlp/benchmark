from __future__ import annotations

import io
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import pywer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from gec_benchmark import load_gec_dataset, normalize_text, score_predictions


SYSTEM_PROMPTS = {
    "gec": """You are a careful grammatical error correction system for Modern Greek.
Correct the user's sentence while preserving its meaning.
Return only the corrected Greek sentence and nothing else.""",
    "toxicity": """You are a Greek text classification system.
Classify the text as offensive or not offensive.
Return only one label: OFF or NOT.""",
    "mt": """You are a translation system.
Translate the user's text accurately.
Return only the translation and nothing else.""",
    "intent": """You are a Greek intent classification system.
Return only the intent label and nothing else.""",
    "summarization": """You are a Greek summarization system.
Produce a concise summary in Greek and return only the summary.""",
}


@dataclass
class TaskSpec:
    name: str
    system_prompt: str
    load_dataset: Callable[..., pd.DataFrame]
    build_prompt: Callable[[dict[str, object]], str]
    evaluate: Callable[[pd.DataFrame], pd.DataFrame]
    normalize_prediction: Callable[[str, dict[str, object]], str] = lambda text, _example: " ".join(text.split())


def _load_dataset_registry(data_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_csv)


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _load_toxicity_dataset(*, data_csv: str | Path, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv, random_state
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run the toxicity benchmark.") from exc

    dataset = load_dataset(
        "csv",
        data_files={
            "train": "https://huggingface.co/datasets/strombergnlp/offenseval_2020/resolve/main/offenseval-gr-training-v1.tsv",
            "test": "https://huggingface.co/datasets/strombergnlp/offenseval_2020/resolve/main/offenseval-gr-test-v1.tsv",
        },
        sep="\t",
    )
    df = pd.DataFrame(dataset[split])
    return df.rename(columns={"subtask_a": "label", "tweet": "text"})[["text", "label"]]


def _load_mt_dataset(
    *,
    data_csv: str | Path,
    random_state: int,
    target_lang: str = "eng",
    split: str = "test",
) -> pd.DataFrame:
    del random_state
    registry = _load_dataset_registry(data_csv)
    repo_url = registry.loc[registry["id"] == 486, "url"].iloc[0]
    zip_bytes = _download_bytes(f"{repo_url}archives/ell-{target_lang}.zip")

    namespace = {"xml": "http://www.w3.org/XML/1998/namespace"}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(f"pgv/ell-{target_lang}.tmx") as tmx_file:
            root = ET.parse(tmx_file).getroot()

    records = []
    for tu in root.findall(".//tu"):
        source = tu.find('.//tuv[@xml:lang="ell"]/seg', namespaces=namespace)
        target = tu.find(f'.//tuv[@xml:lang="{target_lang}"]/seg', namespaces=namespace)
        if source is None or target is None or source.text is None or target.text is None:
            continue
        records.append({"source": source.text, "target": target.text, "target_lang": target_lang})

    df = pd.DataFrame(records).drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    test_df = df.iloc[:500].copy()
    train_df = df.iloc[500:].copy()
    return train_df if split == "train" else test_df


def _load_intent_dataset(*, data_csv: str | Path, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv
    zip_bytes = _download_bytes("https://msensis.com/wp-content/uploads/2023/06/uniway.zip")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        files = {
            "text": "uniway/GR/corpus.txt",
            "ne_tags": "uniway/GR/entities.txt",
            "label": "uniway/GR/intents.txt",
        }
        columns = {}
        for column, inner_path in files.items():
            with zf.open(inner_path) as handle:
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
    return selected.reset_index(drop=True)


def _load_summarization_dataset(*, data_csv: str | Path, random_state: int, split: str = "test") -> pd.DataFrame:
    del data_csv, random_state
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run the summarization benchmark.") from exc

    dataset = load_dataset("DominusTea/GreekLegalSum", split="train")
    df = pd.DataFrame(dataset)
    df["subset"] = df["subset"].astype(int)
    subset_map = {"train": 0, "validation": 1, "test": 2}
    filtered = df.loc[df["subset"] == subset_map[split]].drop(columns=["subset"]).reset_index(drop=True)
    return filtered.rename(columns={"summary": "reference_summary"})


def _build_gec_prompt(example: dict[str, object]) -> str:
    return (
        "Διόρθωσε το παρακάτω ελληνικό κείμενο γραμματικά και ορθογραφικά. "
        "Κράτησε το ίδιο νόημα.\n\n"
        f"Κείμενο: {example['original_text']}\n"
        "Διορθωμένο κείμενο:"
    )


def _build_toxicity_prompt(example: dict[str, object]) -> str:
    return (
        "Classify the following Greek social media text as offensive or not offensive.\n"
        "Return only one label: OFF or NOT.\n\n"
        f"Text: {example['text']}\n"
        "Label:"
    )


def _build_mt_prompt(example: dict[str, object]) -> str:
    return (
        f"Given a text in Greek, translate it to {example['target_lang']}.\n"
        "Only the translation should be returned.\n\n"
        f"Text: {example['source']}\n"
        "Translation:"
    )


def _build_intent_prompt(example: dict[str, object]) -> str:
    labels = example["label_space"]
    return (
        "Given a text, provide the intent of the text. "
        "Only the intent should be returned.\n"
        f"Here is the list of possible intents: {labels}\n\n"
        f"Text: {example['text']}\n"
        "Intent:"
    )


def _build_summarization_prompt(example: dict[str, object]) -> str:
    return (
        "Given a Greek legal text, provide its summary also in Greek. "
        "Only the summary should be returned.\n\n"
        f"Text: {example['text']}\n"
        "Summary:"
    )


def _normalize_gec_prediction(text: str, _example: dict[str, object]) -> str:
    return " ".join(text.split())


def _normalize_toxicity_prediction(text: str, _example: dict[str, object]) -> str:
    normalized = " ".join(text.split()).upper()
    if "OFF" in normalized and "NOT" not in normalized:
        return "OFF"
    if "NOT" in normalized:
        return "NOT"
    return normalized


def _normalize_intent_prediction(text: str, example: dict[str, object]) -> str:
    prediction = " ".join(text.split()).strip()
    valid_labels = [label.strip() for label in str(example["label_space"]).split(",")]
    for label in valid_labels:
        if prediction.lower() == label.lower():
            return label
    return prediction


def _evaluate_gec(raw: pd.DataFrame) -> pd.DataFrame:
    results = raw.copy()
    results["original_text"] = normalize_text(results["original_text"])
    results["corrected_text"] = normalize_text(results["corrected_text"])
    results["prediction"] = normalize_text(results["prediction"])
    return score_predictions(results).assign(task="gec")


def _evaluate_classification(raw: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "task": group["task"].iloc[0],
                "model": model,
                "samples": len(group),
                "accuracy": accuracy_score(group[label_col], group["prediction"]),
                "macro_f1": f1_score(group[label_col], group["prediction"], average="macro", zero_division=0),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=[False, False])


def _best_reference_wer(prediction: str, reference: str) -> float:
    return pywer.wer([reference], [prediction])


def _best_reference_cer(prediction: str, reference: str) -> float:
    return pywer.cer([reference], [prediction])


def _evaluate_mt(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target_lang), group in raw.groupby(["model", "target_lang"], sort=False):
        rows.append(
            {
                "task": f"mt_{target_lang}",
                "model": model,
                "samples": len(group),
                "wer_vs_reference": group.apply(lambda row: _best_reference_wer(row["prediction"], row["target"]), axis=1).mean(),
                "cer_vs_reference": group.apply(lambda row: _best_reference_cer(row["prediction"], row["target"]), axis=1).mean(),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["wer_vs_reference", "cer_vs_reference"], ascending=[True, True])


def _evaluate_summarization(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "task": "summarization",
                "model": model,
                "samples": len(group),
                "wer_vs_reference": group.apply(
                    lambda row: _best_reference_wer(row["prediction"], row["reference_summary"]),
                    axis=1,
                ).mean(),
                "cer_vs_reference": group.apply(
                    lambda row: _best_reference_cer(row["prediction"], row["reference_summary"]),
                    axis=1,
                ).mean(),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["wer_vs_reference", "cer_vs_reference"], ascending=[True, True])


def _load_gec_for_suite(*, data_csv: str | Path, random_state: int) -> pd.DataFrame:
    del random_state
    return load_gec_dataset(data_csv=data_csv)[["original_text", "corrected_text"]]


def _load_intent_with_label_space(*, data_csv: str | Path, random_state: int, split: str = "test") -> pd.DataFrame:
    df = _load_intent_dataset(data_csv=data_csv, random_state=random_state, split=split)
    label_space = ", ".join(sorted(df["label"].unique()))
    df["label_space"] = label_space
    return df


TASKS: dict[str, TaskSpec] = {
    "gec": TaskSpec(
        name="gec",
        system_prompt=SYSTEM_PROMPTS["gec"],
        load_dataset=_load_gec_for_suite,
        build_prompt=_build_gec_prompt,
        normalize_prediction=_normalize_gec_prediction,
        evaluate=_evaluate_gec,
    ),
    "toxicity": TaskSpec(
        name="toxicity",
        system_prompt=SYSTEM_PROMPTS["toxicity"],
        load_dataset=_load_toxicity_dataset,
        build_prompt=_build_toxicity_prompt,
        normalize_prediction=_normalize_toxicity_prediction,
        evaluate=lambda raw: _evaluate_classification(raw, "label"),
    ),
    "mt_eng": TaskSpec(
        name="mt_eng",
        system_prompt=SYSTEM_PROMPTS["mt"],
        load_dataset=lambda **kwargs: _load_mt_dataset(target_lang="eng", **kwargs),
        build_prompt=_build_mt_prompt,
        evaluate=_evaluate_mt,
    ),
    "mt_jpn": TaskSpec(
        name="mt_jpn",
        system_prompt=SYSTEM_PROMPTS["mt"],
        load_dataset=lambda **kwargs: _load_mt_dataset(target_lang="jpn", **kwargs),
        build_prompt=_build_mt_prompt,
        evaluate=_evaluate_mt,
    ),
    "mt_fas": TaskSpec(
        name="mt_fas",
        system_prompt=SYSTEM_PROMPTS["mt"],
        load_dataset=lambda **kwargs: _load_mt_dataset(target_lang="fas", **kwargs),
        build_prompt=_build_mt_prompt,
        evaluate=_evaluate_mt,
    ),
    "intent": TaskSpec(
        name="intent",
        system_prompt=SYSTEM_PROMPTS["intent"],
        load_dataset=_load_intent_with_label_space,
        build_prompt=_build_intent_prompt,
        normalize_prediction=_normalize_intent_prediction,
        evaluate=lambda raw: _evaluate_classification(raw, "label"),
    ),
    "summarization": TaskSpec(
        name="summarization",
        system_prompt=SYSTEM_PROMPTS["summarization"],
        load_dataset=_load_summarization_dataset,
        build_prompt=_build_summarization_prompt,
        evaluate=_evaluate_summarization,
    ),
}


def list_tasks() -> list[str]:
    return sorted(TASKS)
