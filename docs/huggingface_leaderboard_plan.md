# Hugging Face Leaderboard Readiness Plan

This repository is close to leaderboard-ready in terms of evaluation logic, but it is not yet a Hugging Face leaderboard project out of the box. The current stack is centered on local Ollama inference and filesystem-based CSV outputs. A Hugging Face leaderboard usually needs three extra layers:

1. stable published benchmark data
2. a canonical submission/result format
3. a public frontend, usually a Space

This document describes the minimum changes needed to get there while preserving the existing benchmark runner.

## Current Strengths

The repository already has a strong evaluation core:

- unified task execution in [scripts/run_all_benchmarks.py](/Users/iopa3492/vs/benchmark/scripts/run_all_benchmarks.py)
- shared task execution and output generation in [benchmark_suite/core.py](/Users/iopa3492/vs/benchmark/benchmark_suite/core.py)
- deterministic task caps and seeds in [scripts/run_all_benchmarks.py](/Users/iopa3492/vs/benchmark/scripts/run_all_benchmarks.py)
- per-task evaluation logic in [benchmark_suite/task_defs](/Users/iopa3492/vs/benchmark/benchmark_suite/task_defs)
- combined summaries and model-level reporting in [scripts/build_full_benchmark_report.py](/Users/iopa3492/vs/benchmark/scripts/build_full_benchmark_report.py)

That means the benchmark logic itself is already reusable for a leaderboard.

## Main Gaps

### 1. No published leaderboard dataset surface

At the moment, data is loaded from local wrappers and registry files such as:

- [data.csv](/Users/iopa3492/vs/benchmark/data.csv)
- [data_wrapper.py](/Users/iopa3492/vs/benchmark/data_wrapper.py)

For a Hugging Face leaderboard, each benchmark task should have:

- a stable published dataset or dataset config on the Hugging Face Hub
- frozen evaluation splits
- a dataset card describing provenance, fields, and licenses

### 2. No canonical leaderboard submission format

The benchmark currently writes rich CSV outputs such as:

- `{task}_summary.csv`
- `{task}_predictions.csv`
- `all_tasks_summary.csv`
- `all_tasks_predictions.csv`

These are useful, but a leaderboard needs one stable schema for submissions and one stable schema for displayed results.

### 3. No Hub-native inference or submission workflow

Current model identifiers are local Ollama tags, for example in [scripts/run_all_benchmarks.py](/Users/iopa3492/vs/benchmark/scripts/run_all_benchmarks.py):

- `qwen2.5:7b-instruct`
- `aya-expanse:8b`
- `llama3.1:8b-instruct`

A Hugging Face leaderboard usually expects one of these paths:

- direct evaluation from Hugging Face model IDs
- upload of model predictions in a fixed format
- upload of scored result files for manual curation

### 4. No public leaderboard app

There is no Gradio or Spaces app in the repo today. A Hugging Face leaderboard normally exposes:

- sortable model table
- metric explanations
- filters by task and model family
- links to artifacts and methodology

## Minimum Viable Architecture

The least disruptive design is:

1. keep the current benchmark runner
2. add a leaderboard export layer
3. publish frozen benchmark splits to the Hub
4. build a lightweight Hugging Face Space that reads exported leaderboard rows

This avoids rewriting the benchmark around `evaluate` or replacing Ollama execution immediately.

## Recommended Minimal Changes

### Change 1: Freeze and export benchmark splits

Add a script that materializes the current deterministic evaluation sets as files under a versioned directory, for example:

```text
leaderboard_data/
  v1/
    gec_test.parquet
    intent_classification_test.parquet
    legal_classification_test.parquet
    machine_translation_eng_test.parquet
    machine_translation_fas_test.parquet
    machine_translation_jpn_test.parquet
    ner_test.parquet
    pos_test.parquet
    summarization_test.parquet
```

Suggested new script:

- `scripts/export_leaderboard_splits.py`

It should:

- reuse each task loader through the same task registry path
- apply the same `reasonable` caps when requested
- write frozen Parquet or JSONL files
- emit a split manifest with counts, version, seed, and cap profile

Why this matters:

- the leaderboard must evaluate all submissions against the exact same examples
- paper results and leaderboard results stay aligned

### Change 2: Add stable example IDs to all task datasets

The raw prediction CSVs are useful, but leaderboard submissions should key predictions by explicit example ID.

Add a normalization rule so every loaded task dataframe includes:

- `example_id`
- `task`
- optional `target_lang` for MT

This likely belongs near dataset loading in each task definition under [benchmark_suite/task_defs](/Users/iopa3492/vs/benchmark/benchmark_suite/task_defs) or in a shared helper.

Why this matters:

- leaderboard submissions can be validated cleanly
- prediction files become diffable and auditable

### Change 3: Add a canonical leaderboard export

Add a script that converts a finished run into two normalized artifacts:

1. `leaderboard_submission.json`
2. `leaderboard_results.json`

Suggested new script:

- `scripts/export_leaderboard_results.py`

Suggested `leaderboard_results.json` shape:

```json
{
  "benchmark_name": "greek-nlp-benchmark",
  "benchmark_version": "v1",
  "run_id": "20260328_182745_full_suite_default_models_capped500",
  "model_name": "gemma2:9b",
  "backend": "ollama",
  "metrics": {
    "gec": {"gleu_vs_reference": 0.1371},
    "intent_classification": {"macro_f1": 0.7372},
    "legal_classification": {"macro_f1": 0.0115},
    "machine_translation_eng": {"chrf": 61.7807},
    "machine_translation_fas": {"chrf": 17.1234},
    "machine_translation_jpn": {"chrf": 16.7259},
    "ner": {"macro_f1": 0.1218},
    "pos": {"macro_f1": 0.4862},
    "summarization": {"bertscore_f1": 0.5190}
  },
  "aggregate": {
    "avg_normalized_quality": 71.5802
  },
  "metadata": {
    "prompt_version": "v1",
    "cap_profile": "reasonable",
    "temperature": 0.0,
    "num_predict": 256
  }
}
```

Why this matters:

- a Space can read one file format consistently
- submissions do not need to parse multiple internal CSVs

### Change 4: Add a leaderboard-friendly model metadata layer

Right now models are identified by local serving names. Add a small mapping file such as:

- `leaderboard/model_registry.yaml`

Example fields:

- `display_name`
- `ollama_name`
- `huggingface_model_id`
- `parameter_scale`
- `license`
- `chat_template_family`

Why this matters:

- the leaderboard can display user-friendly names
- one evaluated model can be linked back to a public Hub model page

### Change 5: Separate evaluation backend from leaderboard ingestion

Do not block the leaderboard on direct HF inference support.

Phase 1 should accept scored result artifacts generated by this repo.

That means:

- keep [benchmark_suite/core.py](/Users/iopa3492/vs/benchmark/benchmark_suite/core.py) and the Ollama backend
- add a submission ingestion contract on top

Later, Phase 2 can add:

- a Hugging Face `transformers` backend
- optional direct leaderboard evaluation from HF model IDs

This keeps the minimal path small.

### Change 6: Build a small Hugging Face Space

Create a separate Space repo with:

- `app.py`
- `requirements.txt`
- `leaderboard_data/results.jsonl` or equivalent

The app should:

- load normalized leaderboard rows
- display aggregate ranking
- allow per-task filtering
- link to methodology and paper
- show the exact primary metric per task

Gradio is enough for the first version.

### Change 7: Add versioning and validation

Add benchmark version metadata and a validator script:

- `scripts/validate_leaderboard_submission.py`

Validation should check:

- benchmark version exists
- required tasks are present
- metric names are correct
- example IDs match the frozen split
- no duplicated predictions

Why this matters:

- leaderboard data stays clean
- benchmark evolution does not silently mix versions

## Suggested Implementation Order

### Phase 1: Repo changes only

1. add `example_id` to task datasets
2. add `scripts/export_leaderboard_splits.py`
3. add `scripts/export_leaderboard_results.py`
4. add `scripts/validate_leaderboard_submission.py`
5. add `leaderboard/model_registry.yaml`

This phase is enough to make the benchmark output leaderboard-compatible.

### Phase 2: Hub publication

1. publish frozen benchmark splits to a Hugging Face dataset repo
2. publish a dataset card
3. upload initial result rows from existing runs

### Phase 3: Frontend

1. create a Hugging Face Space
2. render aggregate and per-task tables
3. add documentation for how to submit new models

## What Does Not Need To Change Immediately

These pieces can stay as they are for the first leaderboard version:

- local Ollama execution in [benchmark_suite/core.py](/Users/iopa3492/vs/benchmark/benchmark_suite/core.py)
- report generation in [scripts/build_full_benchmark_report.py](/Users/iopa3492/vs/benchmark/scripts/build_full_benchmark_report.py)
- paper asset generation in [scripts/build_publication_assets.py](/Users/iopa3492/vs/benchmark/scripts/build_publication_assets.py)

The leaderboard layer can sit on top of them.

## Bottom Line

This repository is structurally compatible with a Hugging Face leaderboard, but it needs one explicit portability layer.

The smallest credible path is:

1. freeze evaluation splits
2. add example IDs
3. export normalized leaderboard result JSON
4. publish those artifacts to the Hub
5. build a simple Space to display them

That is enough to turn the current benchmark into a maintainable leaderboard without redesigning the evaluation core.
