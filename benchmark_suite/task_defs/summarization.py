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
    median_text_length = int(filtered["text"].fillna("").astype(str).str.len().median())
    print(f"[summarization] truncating input texts to median test length: {median_text_length} characters", flush=True)
    filtered["text"] = filtered["text"].fillna("").astype(str).str.slice(0, median_text_length)
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
        rouge_1 = float("nan")
        rouge_2 = float("nan")
        rouge_l = float("nan")
        bertscore_p = float("nan")
        bertscore_r = float("nan")
        bertscore_f1 = float("nan")

        predictions = group["prediction"].fillna("").astype(str).tolist()
        references = group["reference_summary"].fillna("").astype(str).tolist()

        # Added metrics: ROUGE-1/2/L
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            rouge1_vals, rouge2_vals, rougel_vals = [], [], []
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                rouge1_vals.append(score["rouge1"].fmeasure)
                rouge2_vals.append(score["rouge2"].fmeasure)
                rougel_vals.append(score["rougeL"].fmeasure)
            rouge_1 = float(pd.Series(rouge1_vals).mean())
            rouge_2 = float(pd.Series(rouge2_vals).mean())
            rouge_l = float(pd.Series(rougel_vals).mean())
        except Exception:
            pass

        # Added metric: BERTScore for Greek.
        try:
            from bert_score import score as bertscore_score
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            p_scores, r_scores, f1_scores = bertscore_score(
                predictions,
                references,
                model_type="nlpaueb/bert-base-greek-uncased-v1",
                lang="el",
                verbose=False,
                device=device,
            )
            bertscore_p = float(p_scores.mean().item())
            bertscore_r = float(r_scores.mean().item())
            bertscore_f1 = float(f1_scores.mean().item())
        except Exception:
            pass

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
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "bertscore_precision": bertscore_p,
                "bertscore_recall": bertscore_r,
                "bertscore_f1": bertscore_f1,
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
