from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backends import OllamaBackend
from .tasks import TASKS, TaskSpec


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    num_predict: int = 256
    timeout_seconds: int = 300


def _sample_dataset(dataset: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(dataset):
        return dataset.reset_index(drop=True)
    return dataset.sample(sample_size, random_state=random_state).reset_index(drop=True)


def run_task(
    *,
    task_name: str,
    models: Iterable[str],
    sample_size: int | None = 100,
    random_state: int = 42,
    data_csv: str | Path = "data.csv",
    config: GenerationConfig | None = None,
    backend: OllamaBackend | None = None,
    task_options: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task_name not in TASKS:
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {', '.join(sorted(TASKS))}")

    config = config or GenerationConfig()
    backend = backend or OllamaBackend()
    task_options = task_options or {}
    task: TaskSpec = TASKS[task_name]

    dataset = task.load_dataset(data_csv=data_csv, random_state=random_state, **task_options)
    dataset = _sample_dataset(dataset, sample_size=sample_size, random_state=random_state)

    records: list[dict[str, object]] = []
    for model in models:
        for row in dataset.itertuples(index=False):
            example = row._asdict()
            result = backend.generate(
                model=model,
                prompt=task.build_prompt(example),
                system_prompt=task.system_prompt,
                temperature=config.temperature,
                num_predict=config.num_predict,
                timeout_seconds=config.timeout_seconds,
            )
            record = {
                "task": task_name,
                "model": model,
                "prediction": task.normalize_prediction(result.response, example),
                "latency_seconds": result.latency_seconds,
            }
            record.update(example)
            records.append(record)

    raw = pd.DataFrame.from_records(records)
    summary = task.evaluate(raw)
    return summary, raw


def save_run_outputs(summary: pd.DataFrame, raw: pd.DataFrame, output_dir: str | Path, task_name: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / f"{task_name}_summary.csv", index=False)
    raw.to_csv(output_path / f"{task_name}_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
