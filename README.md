# The Greek NLP Benchmark

This repository benchmarks Greek-capable language models across seven supported NLP tasks: Grammatical Error Correction (GEC), Machine Translation (MT), Intent Classification, Legal Text Classification, Named Entity Recognition (NER), Part-of-Speech (POS) Tagging, and Summarization. It brings together the task datasets, task-specific prompting logic, a unified Python runner, and Colab notebooks for repeated evaluation with Ollama-based local models. The repository also preserves older exploratory notebooks and analyses under `misc/`, but the main runnable benchmark surface is now organized around the current seven-task suite and its task-specific Colab entrypoints.

## Guidelines
* Explore the Greek datasets in [data.csv](data.csv). 
* Use the reorganized Colab notebooks under [notebooks/colab](notebooks/colab).
* Legacy exploratory notebooks now live under [misc/notebooks](misc/notebooks).

## Running The Benchmark Suite
The supported task set is:
* `gec`
* `machine_translation`
* `intent_classification`
* `legal_classification`
* `ner`
* `pos`
* `summarization`

### Setup
1. Create and activate a virtual environment.
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Start Ollama and pull the models you want to compare, for example:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull aya-expanse:8b
ollama pull llama3.1:8b
```

### Unified Python Runner
Use [`scripts/run_all_benchmarks.py`](scripts/run_all_benchmarks.py) to run one task or the whole benchmark suite.

Run all supported tasks:
```bash
python scripts/run_all_benchmarks.py --task all --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b --sample-size 100
```

Run a single task:
```bash
python scripts/run_all_benchmarks.py --task ner --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100
```

Run on the full available dataset for a task:
```bash
python scripts/run_all_benchmarks.py --task summarization --sample-size 0
```

Run repeated Monte Carlo-style sampled evaluations (mean + SEM):
```bash
python scripts/run_all_benchmarks.py --task all --sample-size 100 --repeats 5
```

The compatibility entrypoint [`suite_benchmark.py`](suite_benchmark.py) forwards to the same runner, so this also works:
```bash
python suite_benchmark.py --task all
```

Outputs are written under `results/full_benchmark_suite/`.

For long-running server work, a clearer layout is to keep sampled and full-dataset runs separate, for example:
```bash
results/server_runs/
  completed_runs/
    20260326_235652_full_suite_default_models_sample100_volume_labels/
  full_test_set/
    20260327_XXXXXX_full_suite_default_models_full_test/
```
Pass `--output-dir` explicitly when you want to keep a run in one of these directories.

When `--repeats 1` (default, single run):
* `{task}_summary.csv`
* `{task}_predictions.csv`
* `{task}_visualization.html`
* `all_tasks_summary.csv` when `--task all` is used

When `--repeats > 1` (Monte Carlo mode):
* `{task}/repeat_XX/{task}_summary.csv`
* `{task}/repeat_XX/{task}_predictions.csv`
* `{task}/repeat_XX/{task}_visualization.html`
* `{task}/{task}_summary_with_sem.csv`
* `{task}/{task}_repeat_summaries.csv`
* `{task}/{task}_repeat_predictions.csv`
* `all_tasks_summary_with_sem.csv` when `--task all` is used

Useful flags:
* `--task`: one of `all`, `gec`, `machine_translation`, `intent_classification`, `legal_classification`, `ner`, `pos`, `summarization`
* `--models`: one or more Ollama model names
* `--sample-size`: number of examples to score; use `0` for the full dataset
* `--repeats`: optional; number of repeated sampled runs (default: `1`). Use `>1` with `--sample-size > 0`.
* `--random-state`: sampling seed
* `--output-dir`: where result files are written
* `--temperature`: Ollama sampling temperature
* `--num-predict`: maximum output tokens
* `--timeout-seconds`: request timeout per generation

### Running On A Server
To run the benchmark on a remote Linux server:

1. Clone the repository and enter it:
```bash
git clone https://github.com/greek-nlp/benchmark.git
cd benchmark
```
2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Install Ollama on the server and start it:
```bash
ollama serve
```
5. In another shell, pull the models you want to benchmark:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull aya-expanse:8b
ollama pull llama3.1:8b
```
6. Run the benchmark:
```bash
python scripts/run_all_benchmarks.py --task all --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b --sample-size 100
```

To run a single task on the server:
```bash
python scripts/run_all_benchmarks.py --task ner --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100
```

To keep a long benchmark running after disconnecting, use `tmux` or `screen`. For example:
```bash
tmux new -s benchmark
python scripts/run_all_benchmarks.py --task all --sample-size 100
```

Server outputs are written under:
```bash
results/full_benchmark_suite/
```

## Colab Notebooks
The current Colab entrypoints are:
* [Suite notebook](notebooks/colab/suite/suite_benchmark_colab_monte_carlo.ipynb)
* [Legal classification notebook](notebooks/colab/tasks/legal_classification/legal_text_classification_colab_monte_carlo.ipynb)
* [NER notebook](notebooks/colab/tasks/ner/ner_colab_monte_carlo.ipynb)
* [POS tagging notebook](notebooks/colab/tasks/pos/pos_tagging_colab_monte_carlo.ipynb)

These notebooks follow the same general pattern:
* install dependencies
* start Ollama
* pull selected models
* run repeated Monte Carlo-style evaluations
* save results and zip outputs

## Greek GEC Benchmark In VS Code
To benchmark accessible Greek-capable LLMs for grammatical error correction locally, use [`gec_benchmark.py`](gec_benchmark.py) with Ollama.

1. Create and activate a virtual environment.
2. Install the dependencies:
```bash
pip install pandas pywer zenodo-get wget datasets conll-df openpyxl
```
3. Start Ollama and pull the models you want to compare, for example:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull aya-expanse:8b
ollama pull llama3.1:8b
```
4. Run the benchmark:
```bash
python gec_benchmark.py --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b --sample-size 100
```

The benchmark uses the `KorreDt` dataset, prompts each model to correct Modern Greek text, and writes:
* `results/gec_ollama/gec_benchmark_summary.csv`
* `results/gec_ollama/gec_benchmark_predictions.csv`

## Monte Carlo Runner
For repeated sampled runs with mean and standard error of the mean (SEM), use [`suite_benchmark_monte_carlo.py`](suite_benchmark_monte_carlo.py).

How to run it:

1. Create and activate a virtual environment.
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Start Ollama and pull the models you want to compare, for example:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull aya-expanse:8b
ollama pull llama3.1:8b
```
4. Run one task:
```bash
python suite_benchmark_monte_carlo.py --task ner --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100 --num-splits 5
```
5. Run all supported tasks:
```bash
python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --data-limit-per-task 500 --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b
```

Example:
```bash
python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --data-limit-per-task 500 --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b
```

To resume a long run on a server or after a Colab disconnect:
```bash
python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --resume
```

`--num-splits` controls how many repeated sampled runs are performed per task. `--data-limit-per-task` caps the task dataset before sampling; use `0` to keep the full dataset. The older `--repeats` flag still works as an alias for `--num-splits`.

Useful flags:
* `--task`: run a single task such as `ner`, `gec`, or `summarization`, or use `all`.
* `--models`: one or more Ollama model names.
* `--sample-size`: how many examples to score in each split. Use `0` for the full available dataset after any task cap.
* `--num-splits`: how many repeated sampled runs to perform per task.
* `--data-limit-per-task`: maximum number of examples to keep per task before sampling.
* `--resume`: reuse already saved split outputs instead of recomputing them.

This writes:
* `results/suite_monte_carlo/{task}/repeat_XX/{task}_summary.csv`
* `results/suite_monte_carlo/{task}/{task}_summary_with_sem.csv`
* `results/suite_monte_carlo/all_tasks_summary_with_sem.csv`
* `results/suite_monte_carlo/performance_by_task.csv`

## Requirements
* [zenodo-get](https://github.com/dvolgyes/zenodo_get)
* [datasets](https://pypi.org/project/datasets/)

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
