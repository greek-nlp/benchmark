from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import types

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import datasets  # noqa: F401
except Exception:
    datasets_mod = types.ModuleType("datasets")

    def _missing_load_dataset(*args, **kwargs):
        raise ImportError("datasets is required for this dataset loader.")

    datasets_mod.load_dataset = _missing_load_dataset
    sys.modules["datasets"] = datasets_mod

import data_wrapper as dw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute benchmark dataset sizes per split (raw corpora excluded)."
    )
    parser.add_argument("--data-csv", default="data.csv", help="Path to benchmark dataset registry CSV.")
    parser.add_argument("--output-csv", default="results/dataset_split_sizes.csv", help="Where to save output CSV.")
    return parser.parse_args()


def _safe_len(df) -> int:
    return int(len(df))


def main() -> None:
    args = parse_args()
    datasets = pd.read_csv(args.data_csv)
    rows: list[dict[str, object]] = []

    # Remove stale sparse-checkout folders so loaders can download cleanly.
    for rid in [56, 244, 285, 438]:
        repo_dir = PROJECT_ROOT / f"repo_{rid}"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

    # Benchmark tasks only (raw datasets intentionally excluded).
    task_loaders = {
        "gec": lambda: dw.KorreDt(datasets=datasets),
        "machine_translation": lambda: dw.ProkopidisMtDt(datasets=datasets),
        "intent_classification": lambda: dw.RizouDt(datasets=datasets),
        "legal_classification": lambda: dw.PapaloukasDt(datasets=datasets),
        "ner": lambda: dw.BarziokasDt(datasets=datasets),
        "pos": lambda: dw.ProkopidisUdDt(datasets=datasets),
        "summarization": lambda: dw.KoniarisDt(datasets=datasets),
    }

    for task_name, build_dataset in task_loaders.items():
        try:
            ds = build_dataset()
        except Exception as exc:
            rows.append(
                {
                    "task": task_name,
                    "split": None,
                    "size": None,
                    "status": "error",
                    "error": f"dataset_init_error: {exc}",
                }
            )
            continue

        if task_name == "machine_translation":
            # The benchmark MT task merges eng/jpn/fas rows per split.
            for split in ["train", "test"]:
                try:
                    total = 0
                    for lang in ds.target_langs:
                        total += _safe_len(ds.get(target_lang=lang, split=split))
                    rows.append(
                        {
                            "task": task_name,
                            "split": split,
                            "size": total,
                            "status": "ok",
                            "error": None,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "task": task_name,
                            "split": split,
                            "size": None,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
            continue

        for split in ["train", "validation", "test"]:
            try:
                size = _safe_len(ds.get(split=split))
                rows.append(
                    {
                        "task": task_name,
                        "split": split,
                        "size": size,
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "task": task_name,
                        "split": split,
                        "size": None,
                        "status": "error",
                        "error": str(exc),
                    }
                )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("\n=== Successful sizes ===")
    ok_df = out_df[out_df["status"] == "ok"][["task", "split", "size"]]
    if ok_df.empty:
        print("No successful loads.")
    else:
        print(ok_df.to_string(index=False))

    err_df = out_df[out_df["status"] == "error"][["task", "split", "error"]]
    if not err_df.empty:
        print("\n=== Errors ===")
        print(err_df.to_string(index=False))


if __name__ == "__main__":
    main()
