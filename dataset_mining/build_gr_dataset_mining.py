#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _year_tag(min_year: int, max_year: int) -> str:
    return f"{min_year}_mid{max_year}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Greek NLP dataset-mining CSV end-to-end for a year range.")
    parser.add_argument("--min-year", type=int, required=True)
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--aws-json", default="aws.json")
    parser.add_argument("--model-id", default="meta.llama3-70b-instruct-v1:0")
    parser.add_argument("--output-csv", default=None, help="Final output CSV path.")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--workdir", default="dataset_mining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_year > args.max_year:
        raise ValueError("--min-year must be <= --max-year")

    script_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = _year_tag(args.min_year, args.max_year)

    retrieved = workdir / f".tmp_gr_dataset_mining_{tag}_retrieved.csv"
    screened = workdir / f".tmp_gr_dataset_mining_{tag}_screened.csv"
    heur = workdir / f".tmp_gr_dataset_mining_{tag}_heuristic_urls.csv"
    final = Path(args.output_csv) if args.output_csv else workdir / f"gr_dataset_mining_{tag}_automatic.csv"

    py = sys.executable

    _run(
        [
            py,
            str(script_dir / "retrieve_greek_nlp_papers.py"),
            "--min-year",
            str(args.min_year),
            "--max-year",
            str(args.max_year),
            "--output-csv",
            str(retrieved),
        ]
    )
    _run(
        [
            py,
            str(script_dir / "screen_papers_modern_greek_nlp.py"),
            "--input-csv",
            str(retrieved),
            "--output-csv",
            str(screened),
            "--aws-json",
            args.aws_json,
            "--model-id",
            args.model_id,
        ]
    )
    _run(
        [
            py,
            str(script_dir / "find_dataset_urls.py"),
            "--input-csv",
            str(screened),
            "--output-csv",
            str(heur),
        ]
    )
    _run(
        [
            py,
            str(script_dir / "find_dataset_urls_from_pdf_llm.py"),
            "--input-csv",
            str(heur),
            "--output-csv",
            str(final),
            "--aws-json",
            args.aws_json,
            "--model-id",
            args.model_id,
        ]
    )

    if not args.keep_intermediate:
        for p in [retrieved, screened, heur]:
            if p.exists():
                p.unlink()

    print(f"Done. Final CSV: {final}")


if __name__ == "__main__":
    main()
