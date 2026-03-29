from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing benchmark metrics from saved prediction CSVs.")
    parser.add_argument("--results-dir", required=True, help="Directory containing per-task predictions and summary CSVs.")
    return parser.parse_args()


def _parse_list_like(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [text]


def _best_reference_lists(series: pd.Series) -> list[list[str]]:
    parsed = []
    max_refs = 1
    for value in series.tolist():
        refs = [item for item in _parse_list_like(value) if item.strip()]
        if not refs:
            refs = [""]
        parsed.append(refs)
        max_refs = max(max_refs, len(refs))
    ref_sets = []
    for index in range(max_refs):
        ref_sets.append([refs[index] if index < len(refs) else refs[0] for refs in parsed])
    return ref_sets


def _update_gec(results_dir: Path) -> pd.DataFrame | None:
    predictions_path = results_dir / "gec_predictions.csv"
    summary_path = results_dir / "gec_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        return None

    raw = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    try:
        from nltk.translate.gleu_score import corpus_gleu
    except Exception:
        return summary

    gleu_by_model: dict[str, float] = {}
    for model, group in raw.groupby("model", sort=False):
        references = [[[token for token in str(ref).split()]] for ref in group["corrected_text"].fillna("").tolist()]
        hypotheses = [[token for token in str(pred).split()] for pred in group["prediction"].fillna("").tolist()]
        gleu_by_model[model] = float(corpus_gleu(references, hypotheses))

    summary["gleu_vs_reference"] = summary["model"].map(gleu_by_model)
    summary.to_csv(summary_path, index=False)
    return summary


def _update_mt(results_dir: Path) -> pd.DataFrame | None:
    predictions_path = results_dir / "machine_translation_predictions.csv"
    summary_path = results_dir / "machine_translation_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        return None

    raw = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    try:
        import sacrebleu
    except Exception:
        return summary

    bleu_by_key: dict[tuple[str, str], float] = {}
    chrf_by_key: dict[tuple[str, str], float] = {}
    for (model, target_lang), group in raw.groupby(["model", "target_lang"], sort=False):
        predictions = group["prediction"].fillna("").astype(str).tolist()
        references = _best_reference_lists(group["target"])
        bleu_by_key[(model, target_lang)] = float(sacrebleu.corpus_bleu(predictions, references).score)
        chrf_by_key[(model, target_lang)] = float(sacrebleu.corpus_chrf(predictions, references).score)

    summary["bleu"] = summary.apply(lambda row: bleu_by_key.get((row["model"], row["target_lang"])), axis=1)
    summary["chrf"] = summary.apply(lambda row: chrf_by_key.get((row["model"], row["target_lang"])), axis=1)
    summary.to_csv(summary_path, index=False)
    return summary


def _update_ner(results_dir: Path) -> pd.DataFrame | None:
    predictions_path = results_dir / "ner_predictions.csv"
    summary_path = results_dir / "ner_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        return None

    raw = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    try:
        from seqeval.metrics import f1_score as seqeval_f1_score
        from seqeval.scheme import IOBES
    except Exception:
        return summary

    entity_f1_by_model: dict[str, float] = {}
    for model, group in raw.groupby("model", sort=False):
        gold_sequences = [_parse_list_like(value) for value in group["labels"].tolist()]
        pred_sequences = [_parse_list_like(value) for value in group["prediction"].tolist()]
        pred_sequences = [
            sequence[: len(gold)] + ["O"] * max(0, len(gold) - len(sequence))
            for gold, sequence in zip(gold_sequences, pred_sequences, strict=False)
        ]
        entity_f1_by_model[model] = float(
            seqeval_f1_score(gold_sequences, pred_sequences, mode="strict", scheme=IOBES)
        )

    summary["entity_f1"] = summary["model"].map(entity_f1_by_model)
    summary.to_csv(summary_path, index=False)
    return summary


def _update_summarization(results_dir: Path) -> pd.DataFrame | None:
    predictions_path = results_dir / "summarization_predictions.csv"
    summary_path = results_dir / "summarization_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        return None

    raw = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    try:
        from rouge_score import rouge_scorer
    except Exception:
        return summary

    rouge_by_model: dict[str, dict[str, float]] = {}
    for model, group in raw.groupby("model", sort=False):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        rouge1_vals: list[float] = []
        rouge2_vals: list[float] = []
        rougeL_vals: list[float] = []
        for prediction, reference in zip(group["prediction"].fillna("").astype(str), group["reference_summary"].fillna("").astype(str), strict=False):
            score = scorer.score(reference, prediction)
            rouge1_vals.append(score["rouge1"].fmeasure)
            rouge2_vals.append(score["rouge2"].fmeasure)
            rougeL_vals.append(score["rougeL"].fmeasure)
        rouge_by_model[model] = {
            "rouge_1": float(pd.Series(rouge1_vals).mean()),
            "rouge_2": float(pd.Series(rouge2_vals).mean()),
            "rouge_l": float(pd.Series(rougeL_vals).mean()),
        }

    for column in ["rouge_1", "rouge_2", "rouge_l"]:
        summary[column] = summary["model"].map(lambda model: rouge_by_model.get(model, {}).get(column))
    summary.to_csv(summary_path, index=False)
    return summary


def _rebuild_combined_summary(results_dir: Path) -> None:
    summary_frames = []
    for path in sorted(results_dir.glob("*_summary.csv")):
        if path.name.startswith(("all_tasks_", "report_")):
            continue
        summary_frames.append(pd.read_csv(path))
    if not summary_frames:
        return

    all_columns: list[str] = []
    for frame in summary_frames:
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)

    normalized_frames = [frame.reindex(columns=all_columns) for frame in summary_frames]
    pd.concat(normalized_frames, ignore_index=True).to_csv(results_dir / "all_tasks_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    _update_gec(results_dir)
    _update_mt(results_dir)
    _update_ner(results_dir)
    _update_summarization(results_dir)
    _rebuild_combined_summary(results_dir)

    print(f"Refreshed summaries in {results_dir}")


if __name__ == "__main__":
    main()
