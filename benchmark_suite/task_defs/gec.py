from __future__ import annotations

import pandas as pd
import numpy as np

from gec_benchmark import load_gec_dataset, normalize_text, score_predictions

from ..spec import TaskSpec


SYSTEM_PROMPT = """You are a careful grammatical error correction system for Modern Greek.
Correct the user's sentence while preserving its meaning.
Return only the corrected Greek sentence and nothing else."""


def _load_dataset(*, data_csv: str, random_state: int) -> pd.DataFrame:
    del random_state
    return load_gec_dataset(data_csv=data_csv)[["original_text", "corrected_text"]]


def _build_prompt(example: dict[str, object]) -> str:
    return (
        "Διόρθωσε το παρακάτω ελληνικό κείμενο γραμματικά και ορθογραφικά. "
        "Κράτησε το ίδιο νόημα.\n\n"
        f"Κείμενο: {example['original_text']}\n"
        "Διορθωμένο κείμενο:"
    )


def _normalize_prediction(text: str, _example: dict[str, object]) -> str:
    return " ".join(text.split())


def _evaluate(raw: pd.DataFrame) -> pd.DataFrame:
    results = raw.copy()
    results["original_text"] = normalize_text(results["original_text"])
    results["corrected_text"] = normalize_text(results["corrected_text"])
    results["prediction"] = normalize_text(results["prediction"])
    summary = score_predictions(results).assign(task="gec")

    # Added metric: GLEU (corpus-level) vs reference.
    try:
        from nltk.translate.gleu_score import corpus_gleu
    except Exception:
        summary["gleu_vs_reference"] = np.nan
        return summary

    gleu_by_model = {}
    for model, group in results.groupby("model", sort=False):
        references = [[[tok for tok in ref.split()]] for ref in group["corrected_text"].tolist()]
        hypotheses = [[tok for tok in hyp.split()] for hyp in group["prediction"].tolist()]
        # nltk corpus_gleu expects: list_of_references (per sample: list of refs), hypotheses
        gleu_score = corpus_gleu(references, hypotheses)
        gleu_by_model[model] = float(gleu_score)

    summary["gleu_vs_reference"] = summary["model"].map(gleu_by_model)
    return summary


def build_task() -> TaskSpec:
    return TaskSpec(
        name="gec",
        system_prompt=SYSTEM_PROMPT,
        load_dataset=_load_dataset,
        build_prompt=_build_prompt,
        normalize_prediction=_normalize_prediction,
        evaluate=_evaluate,
    )
