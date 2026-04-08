#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ACL_SOURCE = "acl_anthology"
S2_SOURCE = "semantic_scholar"
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


@dataclass(frozen=True)
class SearchConfig:
    min_year: int = 2024
    max_year: int = 2026
    timeout_seconds: int = 45
    max_s2_results_per_query: int = 2000

    @property
    def years(self) -> set[int]:
        return set(range(self.min_year, self.max_year + 1))


OUTPUT_COLUMNS = [
    "source",
    "paper_id",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "url",
    "doi",
    "query_match_scope",
    "retrieved_at_utc",
]


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _extract_year_acl(paper: Any) -> int | None:
    year_value = getattr(paper, "year", None)
    if year_value is not None:
        try:
            return int(str(year_value))
        except (TypeError, ValueError):
            pass
    full_id = str(getattr(paper, "full_id", ""))
    maybe_year = full_id.split(".", 1)[0]
    if maybe_year.isdigit():
        try:
            return int(maybe_year)
        except ValueError:
            return None
    return None


def _extract_authors_acl(paper: Any) -> str:
    authors = []
    for author in getattr(paper, "authors", []) or []:
        name_obj = getattr(author, "name", None)
        if name_obj is not None:
            first = _norm_text(getattr(name_obj, "first", ""))
            last = _norm_text(getattr(name_obj, "last", ""))
            name = f"{first} {last}".strip()
            if name:
                authors.append(name)
                continue
        authors.append(_norm_text(author))
    return "; ".join(a for a in authors if a)


def _query_scope(title: str, abstract: str, required_term: str = "greek") -> str | None:
    title_l = title.lower()
    abstract_l = abstract.lower()
    in_title = required_term in title_l
    in_abstract = required_term in abstract_l
    if in_title and in_abstract:
        return "title+abstract"
    if in_title:
        return "title"
    if in_abstract:
        return "abstract"
    return None


def fetch_acl(config: SearchConfig, retrieved_at_utc: str) -> list[dict[str, Any]]:
    from acl_anthology import Anthology

    local_repo_cache = os.path.join("dataset_search_2024_2026", ".acl_anthology_cache")
    os.makedirs(local_repo_cache, exist_ok=True)
    anthology = Anthology.from_repo(path=local_repo_cache)
    rows: list[dict[str, Any]] = []

    for paper in anthology.papers():
        year = _extract_year_acl(paper)
        if year is None or year not in config.years:
            continue

        title = _norm_text(getattr(paper, "title", ""))
        abstract = _norm_text(getattr(paper, "abstract", ""))
        scope = _query_scope(title, abstract, required_term="greek")
        if scope is None:
            continue

        rows.append(
            {
                "source": ACL_SOURCE,
                "paper_id": _norm_text(getattr(paper, "full_id", "")),
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": _extract_authors_acl(paper),
                "venue": _norm_text(getattr(paper, "booktitle", "")),
                "url": _norm_text(getattr(paper, "url", "")),
                "doi": _norm_text(getattr(paper, "doi", "")),
                "query_match_scope": scope,
                "retrieved_at_utc": retrieved_at_utc,
            }
        )

    return rows


def _s2_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _extract_authors_s2(paper: dict[str, Any]) -> str:
    names = []
    for author in paper.get("authors", []) or []:
        name = _norm_text(author.get("name", ""))
        if name:
            names.append(name)
    return "; ".join(names)


def _semantic_scholar_query_passes(text_title: str, text_abstract: str) -> bool:
    t = text_title.lower()
    a = text_abstract.lower()
    has_greek = ("greek" in t) or ("greek" in a)
    has_nlp = (
        ("natural language processing" in t)
        or ("natural language processing" in a)
        or (" nlp " in f" {t} ")
        or (" nlp " in f" {a} ")
    )
    return has_greek and has_nlp


def _fetch_s2_query(
    *,
    query: str,
    config: SearchConfig,
    retrieved_at_utc: str,
) -> list[dict[str, Any]]:
    import requests

    params = {
        "query": query,
        "fields": "paperId,title,abstract,year,authors,venue,url,externalIds",
        "year": f"{config.min_year}-{config.max_year}",
        "limit": 1000,
    }
    token: str | None = None
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    while True:
        req_params = dict(params)
        if token:
            req_params["token"] = token

        response = requests.get(S2_ENDPOINT, params=req_params, headers=_s2_headers(), timeout=config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", []) or []

        for paper in data:
            paper_id = _norm_text(paper.get("paperId", ""))
            if not paper_id or paper_id in seen:
                continue

            year = paper.get("year")
            try:
                year_int = int(year)
            except (TypeError, ValueError):
                continue
            if year_int not in config.years:
                continue

            title = _norm_text(paper.get("title", ""))
            abstract = _norm_text(paper.get("abstract", ""))
            if not _semantic_scholar_query_passes(title, abstract):
                continue

            scope = _query_scope(title, abstract, required_term="greek")
            if scope is None:
                continue

            external_ids = paper.get("externalIds", {}) or {}
            doi = _norm_text(external_ids.get("DOI", ""))

            collected.append(
                {
                    "source": S2_SOURCE,
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year_int,
                    "authors": _extract_authors_s2(paper),
                    "venue": _norm_text(paper.get("venue", "")),
                    "url": _norm_text(paper.get("url", "")),
                    "doi": doi,
                    "query_match_scope": scope,
                    "retrieved_at_utc": retrieved_at_utc,
                }
            )
            seen.add(paper_id)

            if len(collected) >= config.max_s2_results_per_query:
                return collected

        token = payload.get("token")
        if not token:
            break

    return collected


def fetch_semantic_scholar(config: SearchConfig, retrieved_at_utc: str) -> list[dict[str, Any]]:
    # Two complementary queries; we deduplicate on paper_id afterwards.
    queries = [
        'greek "natural language processing"',
        "greek nlp",
    ]
    combined: dict[str, dict[str, Any]] = {}
    for query in queries:
        for row in _fetch_s2_query(query=query, config=config, retrieved_at_utc=retrieved_at_utc):
            combined[row["paper_id"]] = row
    return list(combined.values())


def save_csv(rows: list[dict[str, Any]], output_csv: str) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Greek NLP papers from ACL Anthology and Semantic Scholar "
            "for years 2024-2026, with harmonized CSV output."
        )
    )
    parser.add_argument(
        "--output-csv",
        default="dataset_search_2024_2026/greek_nlp_papers_2024_2026.csv",
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["acl", "s2"],
        choices=["acl", "s2"],
        help="Sources to query.",
    )
    parser.add_argument("--min-year", type=int, default=2024, help="Minimum publication year.")
    parser.add_argument("--max-year", type=int, default=2026, help="Maximum publication year.")
    parser.add_argument(
        "--s2-max-results-per-query",
        type=int,
        default=2000,
        help="Maximum Semantic Scholar rows collected per query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SearchConfig(
        min_year=args.min_year,
        max_year=args.max_year,
        max_s2_results_per_query=args.s2_max_results_per_query,
    )
    retrieved_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows: list[dict[str, Any]] = []
    if "acl" in args.sources:
        rows.extend(fetch_acl(config, retrieved_at_utc))
    if "s2" in args.sources:
        rows.extend(fetch_semantic_scholar(config, retrieved_at_utc))

    # Deduplicate by (source, paper_id); keep first.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("source", "")), str(row.get("paper_id", "")))
        if key not in deduped:
            deduped[key] = row

    final_rows = sorted(
        deduped.values(),
        key=lambda r: (str(r.get("source", "")), int(r.get("year", 0)), str(r.get("paper_id", ""))),
    )
    save_csv(final_rows, args.output_csv)

    source_counts: dict[str, int] = {}
    for row in final_rows:
        source = str(row.get("source", ""))
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Saved {len(final_rows)} rows to {args.output_csv}")
    for source, count in sorted(source_counts.items()):
        print(f"  - {source}: {count}")


if __name__ == "__main__":
    main()
