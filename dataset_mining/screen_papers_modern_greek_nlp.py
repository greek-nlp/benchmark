#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL_ID = "meta.llama3-70b-instruct-v1:0"


SCREENING_PROMPT_TEMPLATE = """You are screening academic papers for inclusion in a systematic review.

Inclusion criteria (all must be true to include):
1) The study is about Modern Greek (not Ancient Greek / Koine / historical-only focus).
2) The main modality is textual language (written text NLP; exclude speech-only/audio-only papers).
3) The paper performs NLP (task, method, resource, benchmark, model, annotation, or evaluation in NLP).

Given only the title and abstract below, decide if the paper should be rejected.

Title:
{title}

Abstract:
{abstract}

Return ONLY valid JSON with this exact schema:
{{
  "rejected": true or false,
  "justification": "short reason",
  "mentions_dataset": true or false,
  "dataset_evidence": "short snippet or explanation from title/abstract, or empty string",
  "criteria": {{
    "modern_greek": true or false,
    "textual_modality": true or false,
    "nlp": true or false
  }}
}}
"""


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _is_arxiv_venue(venue: str) -> bool:
    v = _norm_text(venue).lower()
    return "arxiv" in v or "arxiv.org" in v


def _load_aws_credentials(aws_json_path: str) -> dict[str, str]:
    payload = json.loads(Path(aws_json_path).read_text(encoding="utf-8"))
    required = ["aws_access_key_id", "aws_secret_access_key", "aws_region"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Missing keys in {aws_json_path}: {missing}")
    return payload


def _build_bedrock_client(aws_json_path: str):
    import boto3

    creds = _load_aws_credentials(aws_json_path)
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=creds["aws_access_key_id"],
        aws_secret_access_key=creds["aws_secret_access_key"],
        region_name=creds["aws_region"],
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output: {text[:300]}")
    return json.loads(match.group(0))


def _invoke_llama3(
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int = 450,
    temperature: float = 0.0,
) -> str:
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        blocks = resp["output"]["message"]["content"]
        return "\n".join(_norm_text(block.get("text", "")) for block in blocks).strip()
    except Exception:
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return _norm_text(payload.get("generation", ""))


def _default_screening_dict() -> dict[str, Any]:
    return {
        "rejected": True,
        "justification": "No decision available.",
        "mentions_dataset": False,
        "dataset_evidence": "",
        "criteria": {
            "modern_greek": False,
            "textual_modality": False,
            "nlp": False,
        },
    }


def _apply_row_update(row: pd.Series, llm_data: dict[str, Any] | None = None) -> pd.Series:
    out = row.copy()
    venue = _norm_text(out.get("venue", ""))
    excluded_non_peer = _is_arxiv_venue(venue)
    out["excluded_non_peer_reviewed"] = bool(excluded_non_peer)
    out["excluded_non_peer_reviewed_reason"] = (
        "Excluded: venue appears to be arXiv / non peer-reviewed preprint." if excluded_non_peer else ""
    )

    if llm_data is None:
        llm_data = _default_screening_dict()

    criteria = llm_data.get("criteria", {}) or {}
    out["screen_rejected_llm"] = bool(llm_data.get("rejected", True))
    out["screen_rejection_justification"] = _norm_text(llm_data.get("justification", ""))
    out["screen_mentions_dataset"] = bool(llm_data.get("mentions_dataset", False))
    out["screen_dataset_evidence"] = _norm_text(llm_data.get("dataset_evidence", ""))
    out["screen_modern_greek"] = bool(criteria.get("modern_greek", False))
    out["screen_textual_modality"] = bool(criteria.get("textual_modality", False))
    out["screen_nlp"] = bool(criteria.get("nlp", False))
    out["screen_raw_json"] = json.dumps(llm_data, ensure_ascii=False)

    out["excluded_final"] = bool(out["excluded_non_peer_reviewed"] or out["screen_rejected_llm"])
    reasons: list[str] = []
    if out["excluded_non_peer_reviewed"]:
        reasons.append("non_peer_reviewed_arxiv")
    if out["screen_rejected_llm"]:
        reasons.append("fails_scope_screening")
    out["excluded_reasons"] = ";".join(reasons)
    return out


def run_screening(
    input_csv: str,
    output_csv: str,
    aws_json_path: str,
    model_id: str,
    limit: int | None,
    save_every: int,
    skip_llm: bool,
) -> None:
    df = pd.read_csv(input_csv)
    working = df.copy()

    # Ensure columns exist for resumable behavior.
    new_columns = [
        "excluded_non_peer_reviewed",
        "excluded_non_peer_reviewed_reason",
        "screen_rejected_llm",
        "screen_rejection_justification",
        "screen_mentions_dataset",
        "screen_dataset_evidence",
        "screen_modern_greek",
        "screen_textual_modality",
        "screen_nlp",
        "screen_raw_json",
        "excluded_final",
        "excluded_reasons",
    ]
    for col in new_columns:
        if col not in working.columns:
            working[col] = pd.NA

    client = None
    if not skip_llm:
        client = _build_bedrock_client(aws_json_path)

    processed = 0
    for idx, row in working.iterrows():
        if limit is not None and processed >= limit:
            break

        already_done = pd.notna(row.get("excluded_final"))
        if already_done:
            continue

        venue = _norm_text(row.get("venue", ""))
        non_peer = _is_arxiv_venue(venue)
        if non_peer:
            updated = _apply_row_update(row, llm_data={
                "rejected": True,
                "justification": "Excluded as non peer-reviewed arXiv venue.",
                "mentions_dataset": False,
                "dataset_evidence": "",
                "criteria": {
                    "modern_greek": False,
                    "textual_modality": False,
                    "nlp": False,
                },
            })
        else:
            if skip_llm:
                updated = _apply_row_update(row, llm_data=_default_screening_dict())
            else:
                prompt = SCREENING_PROMPT_TEMPLATE.format(
                    title=_norm_text(row.get("title", "")),
                    abstract=_norm_text(row.get("abstract", "")),
                )
                response_text = _invoke_llama3(client=client, model_id=model_id, prompt=prompt)
                llm_json = _extract_json_object(response_text)
                updated = _apply_row_update(row, llm_data=llm_json)

        for col in working.columns:
            working.at[idx, col] = updated.get(col, working.at[idx, col])

        processed += 1
        if processed % save_every == 0:
            working.to_csv(output_csv, index=False)
            print(f"Checkpoint saved after {processed} rows -> {output_csv}")

    working.to_csv(output_csv, index=False)
    print(f"Saved screened CSV to {output_csv}")
    if "excluded_final" in working.columns:
        excluded = int(pd.Series(working["excluded_final"]).fillna(False).astype(bool).sum())
        print(f"Excluded papers: {excluded}/{len(working)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen retrieved Greek NLP papers with rule + LLM criteria.")
    parser.add_argument(
        "--input-csv",
        default="dataset_mining/greek_nlp_papers_2024_2026.csv",
        help="Input papers CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="dataset_mining/greek_nlp_papers_2024_2026_screened.csv",
        help="Output screened CSV.",
    )
    parser.add_argument(
        "--aws-json",
        default="aws.json",
        help="Path to AWS credentials JSON (not committed).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Bedrock model id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows to process (for quick tests).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write checkpoints every N processed rows.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only apply arXiv exclusion (no Bedrock calls).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_screening(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        aws_json_path=args.aws_json,
        model_id=args.model_id,
        limit=args.limit,
        save_every=args.save_every,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
