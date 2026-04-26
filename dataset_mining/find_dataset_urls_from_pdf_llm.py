#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader


URL_RE = re.compile(r"https?://[^\s<>\"]+")
DEFAULT_MODEL_ID = "meta.llama3-70b-instruct-v1:0"


LLM_PROMPT = """You are given a paper title/abstract and a list of URLs extracted from the paper PDF.
Pick the URL that is most likely the dataset/resource link used or released by the paper.

Rules:
- Prefer direct dataset/resource repositories (HuggingFace datasets, Zenodo, Kaggle, GitHub repo with data/resources, institutional dataset pages).
- Reject generic or irrelevant links (ACL Anthology pages, DOI pages, semantic scholar pages, fonts, social media, unrelated websites).
- If none is a plausible dataset/resource link, return empty dataset_url.

Return ONLY valid JSON:
{{
  "dataset_url": "string, empty if none",
  "dataset_name": "string, empty if none",
  "confidence": "high|medium|low",
  "justification": "short reason"
}}

Title: {title}
Abstract: {abstract}
Extracted URLs:
{urls}
"""


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return " ".join(str(v).split()).strip()


def _http_get(url: str, timeout: int) -> tuple[bytes, str, str]:
    req = Request(url, headers={"User-Agent": "dataset-mining/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        content = resp.read()
        content_type = str(resp.headers.get("Content-Type", "")).lower()
        final_url = str(resp.geturl())
    return content, content_type, final_url


def _find_pdf_url(row: pd.Series, timeout: int) -> str:
    source = _norm_text(row.get("source", ""))
    paper_id = _norm_text(row.get("paper_id", ""))
    url = _norm_text(row.get("url", ""))

    if url.lower().endswith(".pdf"):
        return url
    if source == "acl_anthology" and paper_id:
        return f"https://aclanthology.org/{paper_id}.pdf"
    if not url and source == "acl_anthology" and paper_id:
        url = f"https://aclanthology.org/{paper_id}/"
    if not url:
        return ""

    try:
        html_bytes, ctype, final_url = _http_get(url, timeout=timeout)
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            return final_url
        html = html_bytes.decode("utf-8", errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        for href in hrefs:
            full = urljoin(final_url, href).strip()
            low = full.lower()
            if low.endswith(".pdf") or "/pdf/" in low or "paper.pdf" in low:
                return full
    except Exception:
        return ""
    return ""


def _extract_urls_from_pdf(pdf_bytes: bytes, max_pages: int = 40) -> list[str]:
    urls: list[str] = []
    extracted_text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts: list[str] = []
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            texts.append(text)
            for m in URL_RE.finditer(text):
                urls.append(m.group(0))
        extracted_text = "\n".join(texts)
    except Exception:
        return []

    # Also scan raw bytes for URLs that text extraction misses.
    try:
        raw_text = pdf_bytes.decode("latin1", errors="ignore")
        for m in URL_RE.finditer(raw_text):
            urls.append(m.group(0))
    except Exception:
        pass

    # Recover common split URLs from PDF text extraction artifacts.
    if extracted_text:
        urls.extend(_recover_broken_huggingface_urls(extracted_text))

    cleaned = []
    seen = set()
    for u in urls:
        u2 = u.rstrip(").,;:'\"")
        if u2.startswith("http") and u2 not in seen:
            seen.add(u2)
            cleaned.append(u2)
    return cleaned


def _clean_path_fragment(raw: str) -> str:
    tokens = raw.replace("\n", " ").split()
    out: list[str] = []
    stop_words = {"table", "figure", "respectively", "appendix", "section", "http", "https"}
    for tok in tokens:
        t = re.sub(r"[^A-Za-z0-9._-]", "", tok)
        if not t:
            continue
        if out and tok[:1].isupper():
            break
        if t.lower() in stop_words and out:
            break
        out.append(t)
        if len("".join(out)) > 120:
            break
    return "".join(out)


def _recover_broken_huggingface_urls(text: str) -> list[str]:
    recovered: list[str] = []
    compact = text.replace("\u00ad", "")  # soft hyphen

    ds_pattern = re.compile(
        r"https?://\s*hu(?:\s*\d+\s*)?ggingface\.co/\s*datasets/\s*([A-Za-z0-9._-]+)\s*/\s*([A-Za-z0-9._\-\s]{3,180})",
        flags=re.IGNORECASE,
    )
    model_pattern = re.compile(
        r"https?://\s*hu(?:\s*\d+\s*)?ggingface\.co/\s*([A-Za-z0-9._-]+)\s*/\s*([A-Za-z0-9._\-\s]{3,180})",
        flags=re.IGNORECASE,
    )

    for m in ds_pattern.finditer(compact):
        org = re.sub(r"\s+", "", m.group(1))
        name = _clean_path_fragment(m.group(2))
        if org and name:
            recovered.append(f"https://huggingface.co/datasets/{org}/{name}")

    for m in model_pattern.finditer(compact):
        org = re.sub(r"\s+", "", m.group(1))
        name = _clean_path_fragment(m.group(2))
        if org and name:
            recovered.append(f"https://huggingface.co/{org}/{name}")

    # Deduplicate while preserving order.
    out: list[str] = []
    seen = set()
    for u in recovered:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _load_bedrock_client(aws_json_path: str):
    import boto3

    creds = json.loads(Path(aws_json_path).read_text(encoding="utf-8"))
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=creds["aws_access_key_id"],
        aws_secret_access_key=creds["aws_secret_access_key"],
        region_name=creds["aws_region"],
    )


def _invoke_llm(client: Any, model_id: str, prompt: str) -> dict[str, Any]:
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 350, "temperature": 0.0},
    )
    text = "\n".join(_norm_text(block.get("text", "")) for block in resp["output"]["message"]["content"]).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {"dataset_url": "", "dataset_name": "", "confidence": "low", "justification": "Invalid model JSON output."}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"dataset_url": "", "dataset_name": "", "confidence": "low", "justification": "Invalid model JSON output."}


def run(
    input_csv: str,
    output_csv: str,
    aws_json: str,
    model_id: str,
    timeout: int,
    limit: int | None,
    save_every: int,
) -> None:
    df = pd.read_csv(input_csv)
    cols = [
        "pdf_url",
        "pdf_url_found",
        "pdf_text_extracted",
        "pdf_extracted_urls_json",
        "pdf_extracted_url_count",
        "llm_dataset_url",
        "llm_dataset_name",
        "llm_dataset_confidence",
        "llm_dataset_justification",
        "llm_dataset_url_verified",
        "final_dataset_url",
        "final_dataset_url_source",
        "final_dataset_url_confidence",
        "final_dataset_name",
        "final_dataset_url_verified",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    client = _load_bedrock_client(aws_json)

    mask = (df["excluded_final"].fillna(False).astype(bool) == False) & (df["screen_mentions_dataset"].fillna(False).astype(bool))
    idxs = df[mask].index.tolist()
    if limit is not None:
        idxs = idxs[:limit]

    processed = 0
    for idx in idxs:
        row = df.loc[idx]
        if _norm_text(row.get("llm_dataset_url", "")):
            continue

        title = _norm_text(row.get("title", ""))
        abstract = _norm_text(row.get("abstract", ""))
        pdf_url = _find_pdf_url(row, timeout=timeout)
        df.at[idx, "pdf_url"] = pdf_url
        df.at[idx, "pdf_url_found"] = bool(pdf_url)

        extracted_urls: list[str] = []
        text_ok = False
        if pdf_url:
            try:
                pdf_bytes, ctype, final_pdf_url = _http_get(pdf_url, timeout=timeout)
                if final_pdf_url:
                    df.at[idx, "pdf_url"] = final_pdf_url
                if "pdf" in ctype or final_pdf_url.lower().endswith(".pdf") or pdf_bytes[:5] == b"%PDF-":
                    extracted_urls = _extract_urls_from_pdf(pdf_bytes)
                    text_ok = True
            except Exception:
                extracted_urls = []

        df.at[idx, "pdf_text_extracted"] = bool(text_ok)
        df.at[idx, "pdf_extracted_urls_json"] = json.dumps(extracted_urls, ensure_ascii=False)
        df.at[idx, "pdf_extracted_url_count"] = int(len(extracted_urls))

        if extracted_urls:
            prompt = LLM_PROMPT.format(
                title=title,
                abstract=abstract,
                urls="\n".join(f"- {u}" for u in extracted_urls[:150]),
            )
            out = _invoke_llm(client, model_id=model_id, prompt=prompt)
        else:
            out = {"dataset_url": "", "dataset_name": "", "confidence": "low", "justification": "No URLs extracted from PDF."}

        chosen_url = _norm_text(out.get("dataset_url", ""))
        # Enforce user requirement: choose among extracted URLs only.
        if chosen_url and chosen_url not in extracted_urls:
            out["justification"] = _norm_text(out.get("justification", "")) + " | rejected: selected URL not in extracted URL list."
            chosen_url = ""

        verified = False
        if chosen_url:
            try:
                _, _, _ = _http_get(chosen_url, timeout=timeout)
                verified = True
            except Exception:
                verified = False

        df.at[idx, "llm_dataset_url"] = chosen_url
        df.at[idx, "llm_dataset_name"] = _norm_text(out.get("dataset_name", ""))
        df.at[idx, "llm_dataset_confidence"] = _norm_text(out.get("confidence", ""))
        df.at[idx, "llm_dataset_justification"] = _norm_text(out.get("justification", ""))
        df.at[idx, "llm_dataset_url_verified"] = bool(verified)

        processed += 1
        if processed % save_every == 0:
            df.to_csv(output_csv, index=False)
            print(f"Checkpoint saved ({processed}/{len(idxs)}) -> {output_csv}")

    # Merge heuristic and LLM URL candidates into one final URL output.
    for idx, row in df.iterrows():
        llm_url = _norm_text(row.get("llm_dataset_url", ""))
        heur_url = _norm_text(row.get("dataset_url", ""))
        if llm_url:
            df.at[idx, "final_dataset_url"] = llm_url
            df.at[idx, "final_dataset_url_source"] = "pdf_llm"
            df.at[idx, "final_dataset_url_confidence"] = _norm_text(row.get("llm_dataset_confidence", ""))
            df.at[idx, "final_dataset_name"] = _norm_text(row.get("llm_dataset_name", ""))
            df.at[idx, "final_dataset_url_verified"] = bool(row.get("llm_dataset_url_verified", False))
        elif heur_url:
            df.at[idx, "final_dataset_url"] = heur_url
            df.at[idx, "final_dataset_url_source"] = _norm_text(row.get("dataset_url_source", "")) or "heuristic"
            df.at[idx, "final_dataset_url_confidence"] = _norm_text(row.get("dataset_url_confidence", ""))
            df.at[idx, "final_dataset_name"] = _norm_text(row.get("dataset_name", ""))
            df.at[idx, "final_dataset_url_verified"] = bool(row.get("dataset_url_verified", False))
        else:
            df.at[idx, "final_dataset_url"] = ""
            df.at[idx, "final_dataset_url_source"] = ""
            df.at[idx, "final_dataset_url_confidence"] = ""
            df.at[idx, "final_dataset_name"] = ""
            df.at[idx, "final_dataset_url_verified"] = False

    df.to_csv(output_csv, index=False)
    found = df["final_dataset_url"].fillna("").astype(str).str.strip().ne("").sum()
    print(f"Saved -> {output_csv}")
    print(f"Rows in scope: {len(idxs)}")
    print(f"Rows with final_dataset_url: {int(found)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF URL extraction + LLM selection of dataset URL.")
    parser.add_argument("--input-csv", default="dataset_mining/greek_nlp_papers_2024_2026_screened_with_dataset_urls.csv")
    parser.add_argument("--output-csv", default="dataset_mining/greek_nlp_papers_2024_2026_dataset_urls_merged.csv")
    parser.add_argument("--aws-json", default="aws.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        aws_json=args.aws_json,
        model_id=args.model_id,
        timeout=args.timeout,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
