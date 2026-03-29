from __future__ import annotations

import ast
import io
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pywer
from sklearn.metrics import accuracy_score, f1_score


def load_dataset_registry(data_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_csv)


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def normalize_whitespace(value: object) -> str:
    return " ".join(str(value).split()).strip()


def evaluate_classification(raw: pd.DataFrame, *, label_col: str, task_name: str) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model", sort=False):
        labels = group[label_col].fillna("").astype(str).map(normalize_whitespace)
        predictions = group["prediction"].fillna("").astype(str).map(normalize_whitespace)
        rows.append(
            {
                "task": task_name,
                "model": model,
                "samples": len(group),
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)


def best_reference_wer(prediction: str, reference: str | list[str]) -> float:
    if isinstance(reference, list):
        valid_references = [str(item) for item in reference if str(item).strip()]
        if not valid_references:
            return pywer.wer([""], [prediction])
        return min(pywer.wer([candidate], [prediction]) for candidate in valid_references)
    return pywer.wer([reference], [prediction])


def best_reference_cer(prediction: str, reference: str | list[str]) -> float:
    if isinstance(reference, list):
        valid_references = [str(item) for item in reference if str(item).strip()]
        if not valid_references:
            return pywer.cer([""], [prediction])
        return min(pywer.cer([candidate], [prediction]) for candidate in valid_references)
    return pywer.cer([reference], [prediction])


def parse_tmx_records(zip_bytes: bytes, *, target_langs: list[str]) -> pd.DataFrame:
    namespace = {"xml": "http://www.w3.org/XML/1998/namespace"}
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for target_lang in target_langs:
            with archive.open(f"pgv/ell-{target_lang}.tmx") as tmx_file:
                root = ET.parse(tmx_file).getroot()
            for tu in root.findall(".//tu"):
                source = tu.find('.//tuv[@xml:lang="ell"]/seg', namespaces=namespace)
                target = tu.find(f'.//tuv[@xml:lang="{target_lang}"]/seg', namespaces=namespace)
                if source is None or target is None or source.text is None or target.text is None:
                    continue
                records.append({"source": source.text, "target": target.text, "target_lang": target_lang})
    return pd.DataFrame(records)


def parse_label_sequence(text: object, allowed_labels: list[str]) -> list[str]:
    if not isinstance(text, str):
        return []

    normalized = text.strip().replace("\\n", " ")
    parsed: object
    if "[" in normalized and "]" in normalized:
        snippet = normalized[normalized.index("[") : normalized.index("]") + 1]
        if ('"' not in snippet) and ("'" not in snippet):
            for label in allowed_labels:
                snippet = snippet.replace(label, f'"{label}"')
        try:
            parsed = ast.literal_eval(snippet)
        except Exception:
            parsed = []
    else:
        parsed = normalized.replace(",", " ").replace('"', " ").replace("'", " ").split()

    return [normalize_whitespace(item) for item in parsed if normalize_whitespace(item) in allowed_labels]


def align_to_gold(predicted: list[str], gold: list[str], *, filler: str) -> list[str]:
    aligned = list(predicted[: len(gold)])
    if len(aligned) < len(gold):
        aligned.extend(filler for _ in range(len(gold) - len(aligned)))
    return aligned


def evaluate_sequence_labeling(
    raw: pd.DataFrame,
    *,
    task_name: str,
    gold_col: str,
    labels: list[str],
    filler_label: str,
) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model", sort=False):
        gold_flat: list[str] = []
        pred_flat: list[str] = []
        for row in group.itertuples(index=False):
            gold_tags = [normalize_whitespace(tag) for tag in getattr(row, gold_col)]
            pred_tags = align_to_gold(list(getattr(row, "prediction")), gold_tags, filler=filler_label)
            gold_flat.extend(gold_tags)
            pred_flat.extend(pred_tags)
        rows.append(
            {
                "task": task_name,
                "model": model,
                "samples": len(group),
                "tokens": len(gold_flat),
                "accuracy": accuracy_score(gold_flat, pred_flat),
                "macro_f1": f1_score(gold_flat, pred_flat, labels=labels, average="macro", zero_division=0),
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
