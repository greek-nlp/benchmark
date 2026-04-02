from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml


APP_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = APP_ROOT / "data/leaderboard_results.jsonl"
REGISTRY_PATH = APP_ROOT / "data/model_registry.yaml"

TASK_LABELS = {
    "gec": "GEC",
    "intent_classification": "Intent Classification",
    "legal_classification": "Legal Classification",
    "machine_translation_eng": "MT (ENG)",
    "machine_translation_fas": "MT (FAS)",
    "machine_translation_jpn": "MT (JPN)",
    "ner": "NER",
    "pos": "POS",
    "summarization": "Summarization",
}


def load_registry(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    registry: dict[str, dict[str, object]] = {}
    for item in payload.get("models", []):
        if isinstance(item, dict) and isinstance(item.get("ollama_name"), str):
            registry[str(item["ollama_name"])] = item
    return registry


def load_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def available_segments(results: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for result in results:
        primary_metrics = result.get("primary_metrics", {})
        if isinstance(primary_metrics, dict):
            for segment in primary_metrics:
                if segment not in seen:
                    seen.add(segment)
                    ordered.append(segment)
    return ordered


def display_name(model_name: str, registry: dict[str, dict[str, object]]) -> str:
    return str(registry.get(model_name, {}).get("display_name", model_name))


def build_overall_table(results: list[dict[str, object]], registry: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        model_name = str(result.get("model_name", ""))
        aggregate = result.get("aggregate", {})
        registry_entry = registry.get(model_name, {})
        rows.append(
            {
                "Rank": int(aggregate.get("rank", 0)),
                "Model": display_name(model_name, registry),
                "Provider": str(registry_entry.get("provider", "")),
                "Params": str(registry_entry.get("parameter_scale", "")),
                "Avg Quality": round(float(aggregate.get("avg_normalized_quality", 0.0)), 3),
                "Median Quality": round(float(aggregate.get("median_normalized_quality", 0.0)), 3),
                "Avg Latency (s)": round(float(aggregate.get("avg_latency_seconds", 0.0)), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["Rank", "Model"]).reset_index(drop=True)


def build_segment_table(
    results: list[dict[str, object]],
    registry: dict[str, dict[str, object]],
    segment: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        primary = result.get("primary_metrics", {}).get(segment, {})
        metrics = result.get("metrics", {}).get(segment, {})
        if not primary:
            continue
        model_name = str(result.get("model_name", ""))
        rows.append(
            {
                "Model": display_name(model_name, registry),
                "Primary Metric": str(primary.get("metric", "")),
                "Primary Value": round(float(primary.get("value", 0.0)), 4),
                "Samples": int(result.get("samples", {}).get(segment, 0)),
                "Latency (s)": round(float(result.get("aggregate", {}).get("avg_latency_seconds", 0.0)), 3),
                "Metrics": ", ".join(
                    f"{key}={round(float(value), 4) if isinstance(value, (int, float)) else value}"
                    for key, value in metrics.items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["Primary Value", "Model"], ascending=[False, True]).reset_index(drop=True)


REGISTRY = load_registry(REGISTRY_PATH)
RESULTS = load_results(RESULTS_PATH)
SEGMENTS = available_segments(RESULTS)
OVERALL_DF = build_overall_table(RESULTS, REGISTRY)
DEFAULT_SEGMENT = SEGMENTS[0] if SEGMENTS else None
DEFAULT_SEGMENT_DF = build_segment_table(RESULTS, REGISTRY, DEFAULT_SEGMENT) if DEFAULT_SEGMENT else pd.DataFrame()


def on_segment_change(segment: str) -> pd.DataFrame:
    return build_segment_table(RESULTS, REGISTRY, segment)


with gr.Blocks() as demo:
    gr.Markdown("# Greek LLM Leaderboard")
    gr.Markdown(f"Loaded `{len(RESULTS)}` model rows from `{RESULTS_PATH.name}`.")
    gr.Markdown("## Overall Ranking")
    gr.Dataframe(value=OVERALL_DF, interactive=False)
    gr.Markdown("## Per-Segment Results")
    segment_dropdown = gr.Dropdown(
        choices=[(TASK_LABELS.get(segment, segment), segment) for segment in SEGMENTS],
        value=DEFAULT_SEGMENT,
        label="Segment",
    )
    segment_table = gr.Dataframe(value=DEFAULT_SEGMENT_DF, interactive=False)
    if DEFAULT_SEGMENT is not None:
        segment_dropdown.change(on_segment_change, inputs=segment_dropdown, outputs=segment_table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
