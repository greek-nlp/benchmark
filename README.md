# The Greek NLP Benchmark

While large language models (LLMs) have significantly advanced Natural Language Processing (NLP), progress for resource-scarce languages lags behind well-supported ones. Consequently, less-supported languages inherit often non-suited biases and assumptions from well-resourced languages, a gap that is exacerbated by the scarcity of open-access, high-quality language resources. Our work addresses this issue for (Modern) Greek, by collecting and unifying publicly available datasets (created after 2012) into a comprehensive resource. The selected datasets are classified by availability, licensing, and task-specific usage, adhering to the FAIR Data Principles to mitigate biases and to avoid data contamination. We used an open-source (Llama-70b) and a closed-source (GPT-4o mini) LLM to benchmark seven well-known NLP tasks: Toxicity Detection, Grammatical Error Correction, Machine Translation (MT), Summarization, Intent Classification, Named Entity Recognition (NER), Part-Of-Speech (POS) Tagging. Our results show that no model outperforms the other across tasks. Furthermore, we introduce the first benchmark of Text Clustering in Greek, showing that MT (from Greek to English) and Summarization (reducing long texts) along with dense embeddings outperforms TF-IDF when representing long legal documents. Also, we present the first benchmark of Authorship Attribution on publicly available and properly licensed Greek data, revealing the inclusion of authors' works in the LLMs' training data. Repeated at scale, this attribution benchmark can potentially reveal improper data usage. Last, we also assess publicly available, accessible, properly licensed raw data, which could be useful for pre-training purposes.

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
   `pip install -r requirements.txt`
3. Start Ollama and pull the models you want to compare, for example:
   `ollama pull qwen2.5:7b-instruct`
   `ollama pull aya-expanse:8b`
   `ollama pull llama3.1:8b`

### Unified Python Runner
Use [`scripts/run_all_benchmarks.py`](scripts/run_all_benchmarks.py) to run one task or the whole benchmark suite.

Run all supported tasks:
`python scripts/run_all_benchmarks.py --task all --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b --sample-size 100`

Run a single task:
`python scripts/run_all_benchmarks.py --task ner --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100`

Run on the full available dataset for a task:
`python scripts/run_all_benchmarks.py --task summarization --sample-size 0`

The compatibility entrypoint [`suite_benchmark.py`](suite_benchmark.py) forwards to the same runner, so this also works:
`python suite_benchmark.py --task all`

Outputs are written under `results/full_benchmark_suite/`:
* `{task}_summary.csv`
* `{task}_predictions.csv`
* `{task}_visualization.html`
* `all_tasks_summary.csv` when `--task all` is used

Useful flags:
* `--task`: one of `all`, `gec`, `machine_translation`, `intent_classification`, `legal_classification`, `ner`, `pos`, `summarization`
* `--models`: one or more Ollama model names
* `--sample-size`: number of examples to score; use `0` for the full dataset
* `--random-state`: sampling seed
* `--output-dir`: where result files are written
* `--temperature`: Ollama sampling temperature
* `--num-predict`: maximum output tokens
* `--timeout-seconds`: request timeout per generation

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
   `pip install pandas pywer zenodo-get wget datasets conll-df openpyxl`
3. Start Ollama and pull the models you want to compare, for example:
   `ollama pull qwen2.5:7b-instruct`
   `ollama pull aya-expanse:8b`
   `ollama pull llama3.1:8b`
4. Run the benchmark:
   `python gec_benchmark.py --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b --sample-size 100`

The benchmark uses the `KorreDt` dataset, prompts each model to correct Modern Greek text, and writes:
* `results/gec_ollama/gec_benchmark_summary.csv`
* `results/gec_ollama/gec_benchmark_predictions.csv`

## Monte Carlo Runner
For repeated sampled runs with mean and standard error of the mean (SEM), use [`suite_benchmark_monte_carlo.py`](suite_benchmark_monte_carlo.py).

How to run it:

1. Create and activate a virtual environment.
2. Install the dependencies:
   `pip install -r requirements.txt`
3. Start Ollama and pull the models you want to compare, for example:
   `ollama pull qwen2.5:7b-instruct`
   `ollama pull aya-expanse:8b`
   `ollama pull llama3.1:8b`
4. Run one task:
   `python suite_benchmark_monte_carlo.py --task ner --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100 --num-splits 5`
5. Run all supported tasks:
   `python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --data-limit-per-task 500 --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b`

Example:
`python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --data-limit-per-task 500 --models qwen2.5:7b-instruct aya-expanse:8b llama3.1:8b`

To resume a long run on a server or after a Colab disconnect:
`python suite_benchmark_monte_carlo.py --task all --sample-size 100 --num-splits 5 --resume`

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
