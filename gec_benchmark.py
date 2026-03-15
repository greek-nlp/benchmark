from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request

import pandas as pd
import pywer


DEFAULT_MODELS = [
    "qwen2.5:7b-instruct",
    "aya-expanse:8b",
    "llama3.1:8b",
]


SYSTEM_PROMPT = """You are a careful grammatical error correction system for Modern Greek.
Correct the user's sentence while preserving its meaning.
Return only the corrected Greek sentence and nothing else."""


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    num_predict: int = 256
    timeout_seconds: int = 300


def _run_git_command(command: list[str], cwd: str | Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _download_korre_repo(repo_url: str, root_dir: Path) -> Path:
    repo_dir = root_dir / "repo_244"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    _run_git_command(["git", "init", str(repo_dir)])
    _run_git_command(["git", "remote", "add", "-f", "origin", repo_url], cwd=repo_dir)
    _run_git_command(["git", "config", "core.sparseCheckout", "true"], cwd=repo_dir)
    sparse_checkout = repo_dir / ".git" / "info" / "sparse-checkout"
    sparse_checkout.write_text("GNC\n", encoding="utf-8")
    _run_git_command(["git", "pull", "origin", "main"], cwd=repo_dir)
    return repo_dir / "GNC"


def _load_korre_from_repo(repo_url: str, root_dir: Path) -> pd.DataFrame:
    dataset_dir = _download_korre_repo(repo_url, root_dir)
    try:
        df_ann_a = pd.read_excel(dataset_dir / "GNC_annotator_A.xlsx")
        df_ann_a.columns = [
            "label_annA",
            "original_text_annA",
            "corrected_text_annA",
            "error_description_annA",
            "error_type_annA",
            "fluency_annA",
        ]
        df_ann_b = pd.read_excel(dataset_dir / "GNC_annotator_B.xlsx")
        df_ann_b.columns = [
            "label_annB",
            "original_text_annB",
            "corrected_text_annB",
            "error_description_annB",
            "error_type_annB",
            "fluency_annB",
        ]
        annotations = pd.merge(df_ann_a, df_ann_b, left_index=True, right_index=True, how="inner")

        original_lines = (dataset_dir / "orig.txt").read_text(encoding="utf-8").splitlines()
        corrected_lines = (dataset_dir / "corr.txt").read_text(encoding="utf-8").splitlines()
        original = pd.DataFrame({"original_text": original_lines})
        corrected = pd.DataFrame({"corrected_text": corrected_lines})
        text_pairs = pd.merge(original, corrected, left_index=True, right_index=True, how="inner")
        text_pairs = text_pairs.replace("", pd.NA)

        dataset = pd.merge(text_pairs, annotations, left_index=True, right_index=True, how="inner")
        dataset = dataset.drop(
            columns=[
                "original_text_annA",
                "original_text_annB",
                "corrected_text_annA",
                "corrected_text_annB",
            ]
        )
        dataset = dataset.dropna(subset=["original_text", "corrected_text"])
        dataset = dataset.loc[dataset["corrected_text"] != dataset["original_text"]]
        return dataset[["original_text", "corrected_text"]].reset_index(drop=True)
    finally:
        repo_dir = root_dir / "repo_244"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)


def load_gec_dataset(data_csv: str | Path = "data.csv") -> pd.DataFrame:
    datasets = pd.read_csv(data_csv)
    repo_url = datasets.loc[datasets["id"] == 244, "url"].iloc[0]
    return _load_korre_from_repo(repo_url=repo_url, root_dir=Path.cwd())


def build_prompt(text: str) -> str:
    return (
        "Διόρθωσε το παρακάτω ελληνικό κείμενο γραμματικά και ορθογραφικά. "
        "Κράτησε το ίδιο νόημα.\n\n"
        f"Κείμενο: {text}\n"
        "Διορθωμένο κείμενο:"
    )


def call_ollama(model: str, prompt: str, config: GenerationConfig) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://127.0.0.1:11434. "
            "Start Ollama first, then pull the models you want to benchmark."
        ) from exc

    return body["response"].strip()


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def score_predictions(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in results.groupby("model", sort=False):
        original = group["original_text"].tolist()
        reference = group["corrected_text"].tolist()
        prediction = group["prediction"].tolist()

        exact_match = sum(pred == ref for pred, ref in zip(prediction, reference)) / len(group)
        changed_input = sum(pred != src for pred, src in zip(prediction, original)) / len(group)

        rows.append(
            {
                "model": model,
                "samples": len(group),
                "exact_match": exact_match,
                "wer_vs_reference": pywer.wer(reference, prediction),
                "cer_vs_reference": pywer.cer(reference, prediction),
                "wer_vs_original": pywer.wer(original, prediction),
                "cer_vs_original": pywer.cer(original, prediction),
                "changed_input_rate": changed_input,
                "avg_latency_seconds": group["latency_seconds"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["wer_vs_reference", "cer_vs_reference", "exact_match"], ascending=[True, True, False])


def benchmark_ollama_models(
    models: Iterable[str],
    sample_size: int | None = 100,
    random_state: int = 42,
    data_csv: str | Path = "data.csv",
    config: GenerationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or GenerationConfig()
    dataset = load_gec_dataset(data_csv=data_csv)
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.sample(sample_size, random_state=random_state).reset_index(drop=True)

    dataset["original_text"] = normalize_text(dataset["original_text"])
    dataset["corrected_text"] = normalize_text(dataset["corrected_text"])

    records: list[dict[str, object]] = []
    for model in models:
        for row in dataset.itertuples(index=False):
            prompt = build_prompt(row.original_text)
            started = time.perf_counter()
            prediction = call_ollama(model=model, prompt=prompt, config=config)
            latency = time.perf_counter() - started
            records.append(
                {
                    "model": model,
                    "original_text": row.original_text,
                    "corrected_text": row.corrected_text,
                    "prediction": " ".join(prediction.split()),
                    "latency_seconds": latency,
                }
            )

    raw = pd.DataFrame.from_records(records)
    summary = score_predictions(raw)
    return summary, raw


def save_results(summary: pd.DataFrame, raw: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / "gec_benchmark_summary.csv", index=False)
    raw.to_csv(output_path / "gec_benchmark_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Greek GEC with local Ollama models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model names to benchmark.")
    parser.add_argument("--sample-size", type=int, default=100, help="How many Korre examples to score.")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--data-csv", default="data.csv", help="Path to the benchmark dataset registry CSV.")
    parser.add_argument("--output-dir", default="results/gec_ollama", help="Where to save benchmark CSV files.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to Ollama.")
    parser.add_argument("--num-predict", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-request timeout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, raw = benchmark_ollama_models(
        models=args.models,
        sample_size=None if args.sample_size <= 0 else args.sample_size,
        random_state=args.random_state,
        data_csv=args.data_csv,
        config=GenerationConfig(
            temperature=args.temperature,
            num_predict=args.num_predict,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    save_results(summary, raw, args.output_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not math.isnan(x) else "nan"))
    print(f"\nSaved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
