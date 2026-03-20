from __future__ import annotations

import csv
import html
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


def _metric_direction(column: str) -> str:
    lower_is_better_keywords = ("wer", "cer", "latency", "error")
    if any(keyword in column for keyword in lower_is_better_keywords):
        return "lower"
    return "higher"


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_label(column: str) -> str:
    return column.replace("_", " ").title()


def _build_metric_card(summary: pd.DataFrame, column: str) -> str:
    if column not in summary or not pd.api.types.is_numeric_dtype(summary[column]):
        return ""

    clean = summary[["model", column]].dropna()
    if clean.empty:
        return ""

    direction = _metric_direction(column)
    values = clean[column].astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    rows: list[str] = []

    for row in clean.itertuples(index=False):
        value = float(getattr(row, column))
        if span == 0:
            percent = 100.0
        elif direction == "lower":
            percent = ((maximum - value) / span) * 100.0
        else:
            percent = ((value - minimum) / span) * 100.0

        rows.append(
            """
            <div class="metric-row">
              <div class="metric-row-header">
                <span class="model-name">{model}</span>
                <span class="metric-value">{value}</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {percent:.1f}%"></div>
              </div>
            </div>
            """.format(
                model=html.escape(str(row.model)),
                value=html.escape(_format_metric(value)),
                percent=percent,
            ).strip()
        )

    return """
    <section class="card">
      <h2>{label}</h2>
      <p class="metric-note">{note}</p>
      <div class="metric-rows">
        {rows}
      </div>
    </section>
    """.format(
        label=html.escape(_metric_label(column)),
        note="Lower is better." if direction == "lower" else "Higher is better.",
        rows="\n".join(rows),
    ).strip()


def _build_summary_table(summary: pd.DataFrame) -> str:
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in summary.columns)
    rows: list[str] = []
    for row in summary.itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape(_format_metric(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return """
    <section class="card">
      <h2>Summary Table</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{headers}</tr></thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    """.format(headers=headers, rows="\n".join(rows)).strip()


def save_task_visualization(summary: pd.DataFrame, output_dir: str | Path, task_name: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = [
        column
        for column in summary.columns
        if column not in {"task", "model"} and pd.api.types.is_numeric_dtype(summary[column])
    ]
    metric_cards = "\n".join(_build_metric_card(summary, column) for column in numeric_columns)
    report_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      <style>
        :root {{
          color-scheme: light;
          --bg: #f5f1e8;
          --panel: rgba(255, 252, 245, 0.9);
          --panel-border: #d7cab5;
          --text: #1f2933;
          --muted: #52606d;
          --accent: linear-gradient(90deg, #d97706, #0f766e);
          --track: #e6dccd;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(217, 119, 6, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 28%),
            var(--bg);
        }}
        main {{
          max-width: 1100px;
          margin: 0 auto;
          padding: 48px 20px 64px;
        }}
        h1, h2 {{
          margin: 0;
          font-weight: 700;
        }}
        p {{
          margin: 0;
          line-height: 1.5;
        }}
        .hero {{
          margin-bottom: 24px;
        }}
        .hero p {{
          margin-top: 10px;
          color: var(--muted);
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 16px;
          margin-bottom: 16px;
        }}
        .card {{
          background: var(--panel);
          backdrop-filter: blur(8px);
          border: 1px solid var(--panel-border);
          border-radius: 18px;
          padding: 20px;
          box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
        }}
        .metric-note {{
          margin-top: 8px;
          color: var(--muted);
          font-size: 0.95rem;
        }}
        .metric-rows {{
          margin-top: 18px;
          display: grid;
          gap: 14px;
        }}
        .metric-row-header {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 7px;
          font-size: 0.98rem;
        }}
        .model-name {{
          font-weight: 700;
          overflow-wrap: anywhere;
        }}
        .metric-value {{
          color: var(--muted);
          white-space: nowrap;
        }}
        .bar-track {{
          height: 12px;
          border-radius: 999px;
          background: var(--track);
          overflow: hidden;
        }}
        .bar-fill {{
          height: 100%;
          border-radius: inherit;
          background: var(--accent);
        }}
        .table-wrap {{
          margin-top: 16px;
          overflow-x: auto;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
          font-size: 0.92rem;
        }}
        th, td {{
          text-align: left;
          padding: 10px 12px;
          border-bottom: 1px solid var(--panel-border);
          white-space: nowrap;
        }}
        th {{
          color: var(--muted);
          font-weight: 700;
        }}
      </style>
    </head>
    <body>
      <main>
        <section class="hero">
          <h1>{heading}</h1>
          <p>Generated from the task summary. Each chart compares models on one metric for this benchmark run.</p>
        </section>
        <div class="grid">
          {metric_cards}
        </div>
        {summary_table}
      </main>
    </body>
    </html>
    """.format(
        title=html.escape(f"{task_name} benchmark visualization"),
        heading=html.escape(f"{task_name} benchmark visualization"),
        metric_cards=metric_cards,
        summary_table=_build_summary_table(summary),
    ).strip()

    report_path = output_path / f"{task_name}_visualization.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path


def save_run_outputs(summary: pd.DataFrame, raw: pd.DataFrame, output_dir: str | Path, task_name: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / f"{task_name}_summary.csv", index=False)
    raw.to_csv(output_path / f"{task_name}_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    save_task_visualization(summary, output_path, task_name)
