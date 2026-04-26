#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests


URL_RE = re.compile(r"https?://[^\s<>\"]+")
DATASET_KEYWORDS = (
    "dataset",
    "data set",
    "corpus",
    "benchmark",
    "resource",
    "resources",
    "annotations",
    "shared task",
)
PAPER_DOMAINS = (
    "aclanthology.org",
    "arxiv.org",
    "semanticscholar.org",
    "doi.org",
)
GENERIC_BLOCKLIST = (
    "github.com/acl-org/acl-anthology",
    "github.com/acl-org/",
    "aclanthology.org/",
    "fonts.googleapis.com/",
    "fonts.gstatic.com/",
    "semanticscholar.org/search",
    "semanticscholar.org/paper/",
    "google.com/search",
)


@dataclass
class CandidateURL:
    url: str
    source: str
    evidence: str
    score: int
    verified: bool
    host: str


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).split()).strip()


def _clean_url(url: str) -> str:
    url = _norm_text(url)
    url = url.rstrip(").,;:'\"")
    return url


def _extract_urls(text: str) -> list[str]:
    urls = [_clean_url(m.group(0)) for m in URL_RE.finditer(text or "")]
    return [u for u in urls if u.startswith("http")]


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_dataset_host(url: str) -> bool:
    u = url.lower()
    h = _host(url)
    if "huggingface.co" in h and "/datasets/" in u:
        return True
    return any(
        domain in h
        for domain in [
            "zenodo.org",
            "kaggle.com",
            "figshare.com",
            "dataverse",
            "osf.io",
            "github.com",
            "gitlab.com",
            "drive.google.com",
            "dropbox.com",
            "huggingface.co",
        ]
    )


def _url_path_has_dataset_cue(url: str) -> bool:
    u = url.lower()
    cues = ["dataset", "data", "corpus", "benchmark", "resources", "shared-task", "leaderboard"]
    return any(c in u for c in cues)


def _is_generic_blocked_url(url: str) -> bool:
    u = _clean_url(url).lower()
    return any(token in u for token in GENERIC_BLOCKLIST)


def _looks_like_paper_link(url: str) -> bool:
    h = _host(url)
    return any(domain in h for domain in PAPER_DOMAINS)


def _fetch_text(url: str, timeout: int) -> tuple[str, str]:
    resp = requests.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "dataset-mining-bot/1.0"})
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    return resp.text if "text" in content_type or "html" in content_type else "", resp.url


def _find_links_from_html(html: str, base_url: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a"):
            href = _norm_text(a.get("href", ""))
            text = _norm_text(a.get_text(" ", strip=True))
            if not href:
                continue
            full = _clean_url(urljoin(base_url, href))
            if full.startswith("http"):
                links.append((full, text))
        return links
    except Exception:
        pass

    for m in re.finditer(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE):
        href = _clean_url(m.group(1))
        full = _clean_url(urljoin(base_url, href))
        if full.startswith("http"):
            links.append((full, ""))
    return links


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages[:25]:
            texts.append(_norm_text(page.extract_text() or ""))
        return "\n".join(texts)
    except Exception:
        return ""


def _verify_url(url: str, timeout: int) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "dataset-mining-bot/1.0"})
        if r.status_code < 400:
            return True
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "dataset-mining-bot/1.0"})
        return r.status_code < 400
    except Exception:
        return False


def _score_candidate(url: str, source: str, evidence: str, verified: bool = False) -> int:
    score = 0
    low_evidence = (evidence or "").lower()
    if _is_dataset_host(url):
        score += 5
    if _url_path_has_dataset_cue(url):
        score += 3
    if any(k in low_evidence for k in DATASET_KEYWORDS):
        score += 3
    if source in {"pdf", "external_link"}:
        score += 2
    elif source == "landing_page":
        score += 1
    if verified:
        score += 2
    if _looks_like_paper_link(url):
        score -= 3
    if _is_generic_blocked_url(url):
        score -= 8
    return score


def _dataset_name_from_url(url: str) -> str:
    u = _clean_url(url)
    if "huggingface.co/datasets/" in u:
        return u.split("huggingface.co/datasets/")[-1].strip("/")
    parts = [p for p in urlparse(u).path.split("/") if p]
    return parts[-1] if parts else ""


def _confidence(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 5:
        return "medium"
    return "low"


def find_dataset_urls_for_row(row: pd.Series, timeout: int = 20) -> tuple[str, str, str, str, bool, str, str]:
    title = _norm_text(row.get("title", ""))
    abstract = _norm_text(row.get("abstract", ""))
    paper_url = _norm_text(row.get("url", ""))
    source = _norm_text(row.get("source", ""))
    paper_id = _norm_text(row.get("paper_id", ""))
    if not paper_url and source == "acl_anthology" and paper_id:
        paper_url = f"https://aclanthology.org/{paper_id}/"

    candidates: list[CandidateURL] = []
    seen: set[str] = set()

    def add_candidate(url: str, source: str, evidence: str) -> None:
        cu = _clean_url(url)
        if not cu or cu in seen:
            return
        # Skip known generic non-dataset links.
        if _is_generic_blocked_url(cu):
            return
        seen.add(cu)
        score = _score_candidate(cu, source=source, evidence=evidence, verified=False)
        candidates.append(
            CandidateURL(
                url=cu,
                source=source,
                evidence=_norm_text(evidence),
                score=score,
                verified=False,
                host=_host(cu),
            )
        )

    # 1) From abstract/title directly.
    for u in _extract_urls(f"{title}\n{abstract}"):
        add_candidate(u, "abstract", "URL found in title/abstract.")

    fetched_html = ""
    resolved_paper_url = paper_url
    page_links: list[tuple[str, str]] = []

    # 2) Landing page.
    if paper_url:
        try:
            fetched_html, resolved_paper_url = _fetch_text(paper_url, timeout=timeout)
            page_links = _find_links_from_html(fetched_html, resolved_paper_url)
            for link_url, link_text in page_links:
                context = f"anchor_text={link_text}"
                low = f"{link_url.lower()} {link_text.lower()}"
                same_domain = _host(link_url) == _host(resolved_paper_url)
                has_dataset_signal = (
                    any(k in low for k in DATASET_KEYWORDS)
                    or "huggingface.co/datasets/" in low
                    or _url_path_has_dataset_cue(link_url)
                    or ("github.com" in low and "acl-org/acl-anthology" not in low)
                )
                if has_dataset_signal and not (same_domain and "github.com" not in low and "huggingface.co" not in low):
                    add_candidate(link_url, "landing_page", f"Potential dataset link on landing page ({context}).")
        except Exception:
            pass

    # 3) PDF scan (non-OCR).
    pdf_url = ""
    for link_url, link_text in page_links:
        low = f"{link_url.lower()} {link_text.lower()}"
        if link_url.lower().endswith(".pdf") or "/pdf/" in low or " pdf" in low:
            pdf_url = link_url
            break
    if not pdf_url and resolved_paper_url and resolved_paper_url.lower().endswith(".pdf"):
        pdf_url = resolved_paper_url
    if not pdf_url and "aclanthology.org/" in paper_url:
        pdf_url = paper_url.rstrip("/") + ".pdf"

    if pdf_url:
        try:
            resp = requests.get(pdf_url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "dataset-mining-bot/1.0"})
            if resp.status_code < 400:
                pdf_text = _extract_pdf_text(resp.content)
                for u in _extract_urls(pdf_text):
                    add_candidate(u, "pdf", "URL extracted from PDF text.")
        except Exception:
            pass

    # 4) Selected external links likely to hold resources.
    external_scan_budget = 3
    scanned = 0
    for link_url, link_text in page_links:
        if scanned >= external_scan_budget:
            break
        low = f"{link_url.lower()} {link_text.lower()}"
        if _is_generic_blocked_url(link_url):
            continue
        if not any(k in low for k in ["github.com", "dataset", "data", "resources", "supplement", "benchmark", "corpus"]):
            continue
        try:
            html, final_url = _fetch_text(link_url, timeout=timeout)
            scanned += 1
            for u in _extract_urls(html):
                add_candidate(u, "external_link", f"Found while scanning external page: {final_url}")
        except Exception:
            continue

    if not candidates:
        return "", "", "", "", False, "", "No candidate URLs found."

    # Keep likely dataset links only; if none are plausible, return empty.
    likely = [
        c
        for c in candidates
        if (
            _is_dataset_host(c.url)
            or _url_path_has_dataset_cue(c.url)
            or any(k in c.evidence.lower() for k in DATASET_KEYWORDS)
        )
    ]
    if not likely:
        return "", "", "", "", False, "", "No plausible dataset/resource links after filtering."

    ranked = sorted(likely, key=lambda c: c.score, reverse=True)
    # Verify only top few to avoid very slow runs.
    for cand in ranked[:3]:
        cand.verified = _verify_url(cand.url, timeout=timeout)
        cand.score = _score_candidate(cand.url, source=cand.source, evidence=cand.evidence, verified=cand.verified)
    ranked = sorted(ranked, key=lambda c: (c.score, int(c.verified)), reverse=True)
    best = ranked[0]
    host = best.host
    dataset_name = _dataset_name_from_url(best.url)
    notes = f"candidates={len(candidates)}; likely={len(likely)}; best_score={best.score}"
    return (
        best.url,
        best.source,
        _confidence(best.score),
        dataset_name,
        bool(best.verified),
        host,
        f"{best.evidence} | {notes}",
    )


def run(input_csv: str, output_csv: str, timeout: int, limit: int | None, save_every: int) -> None:
    df = pd.read_csv(input_csv)

    new_cols = [
        "dataset_url",
        "dataset_url_source",
        "dataset_url_confidence",
        "dataset_name",
        "dataset_host",
        "dataset_url_verified",
        "dataset_url_evidence",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = pd.NA

    mask = (df["excluded_final"].fillna(False).astype(bool) == False) & (df["screen_mentions_dataset"].fillna(False).astype(bool))
    targets = df[mask].index.tolist()
    if limit is not None:
        targets = targets[:limit]

    processed = 0
    for idx in targets:
        row = df.loc[idx]
        # Skip already enriched rows.
        if _norm_text(row.get("dataset_url", "")):
            continue
        url, source, conf, name, verified, host, evidence = find_dataset_urls_for_row(row, timeout=timeout)
        df.at[idx, "dataset_url"] = url
        df.at[idx, "dataset_url_source"] = source
        df.at[idx, "dataset_url_confidence"] = conf
        df.at[idx, "dataset_name"] = name
        df.at[idx, "dataset_host"] = host
        df.at[idx, "dataset_url_verified"] = bool(verified)
        df.at[idx, "dataset_url_evidence"] = evidence
        processed += 1
        if processed % save_every == 0:
            df.to_csv(output_csv, index=False)
            print(f"Checkpoint saved ({processed}/{len(targets)}) -> {output_csv}")

    df.to_csv(output_csv, index=False)
    found = df["dataset_url"].fillna("").astype(str).str.strip().ne("").sum()
    in_scope = len(targets)
    print(f"Saved -> {output_csv}")
    print(f"Rows in scope: {in_scope}")
    print(f"Rows with dataset_url: {int(found)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find dataset URLs for screened-in papers mentioning datasets.")
    parser.add_argument(
        "--input-csv",
        default="dataset_mining/greek_nlp_papers_2024_2026_screened.csv",
        help="Screened CSV input.",
    )
    parser.add_argument(
        "--output-csv",
        default="dataset_mining/greek_nlp_papers_2024_2026_screened_with_dataset_urls.csv",
        help="Output CSV with dataset URL enrichment.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout per request in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows in scope.")
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint frequency.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        timeout=args.timeout,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
