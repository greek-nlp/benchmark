# The Greek NLP Benchmark

While large language models (LLMs) have significantly advanced Natural Language Processing (NLP), progress for resource-scarce languages lags behind well-supported ones. Consequently, less-supported languages inherit often non-suited biases and assumptions from well-resourced languages, a gap that is exacerbated by the scarcity of open-access, high-quality language resources. Our work addresses this issue for (Modern) Greek, by collecting and unifying publicly available datasets (created after 2012) into a comprehensive resource. The selected datasets are classified by availability, licensing, and task-specific usage, adhering to the FAIR Data Principles to mitigate biases and to avoid data contamination. We used an open-source (Llama-70b) and a closed-source (GPT-4o mini) LLM to benchmark seven well-known NLP tasks: Toxicity Detection, Grammatical Error Correction, Machine Translation (MT), Summarization, Intent Classification, Named Entity Recognition (NER), Part-Of-Speech (POS) Tagging. Our results show that no model outperforms the other across tasks. Furthermore, we introduce the first benchmark of Text Clustering in Greek, showing that MT (from Greek to English) and Summarization (reducing long texts) along with dense embeddings outperforms TF-IDF when representing long legal documents. Also, we present the first benchmark of Authorship Attribution on publicly available and properly licensed Greek data, revealing the inclusion of authors' works in the LLMs' training data. Repeated at scale, this attribution benchmark can potentially reveal improper data usage. Last, we also assess publicly available, accessible, properly licensed raw data, which could be useful for pre-training purposes.

## Guidelines
* Explore the Greek datasets in [data.csv](data.csv). 
* Run [the data notebook](nlp_gr_access_data.ipynb) to download the data.
* Run the experiments for [the task benchmarks](nlp_gr_experiments.ipynb) for the open-source Llama model. 

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

## Shared Benchmark Runner
For a reusable benchmark runner across tasks, use [`suite_benchmark.py`](suite_benchmark.py).

Currently supported tasks:
* `gec`
* `toxicity`
* `mt_eng`
* `mt_jpn`
* `mt_fas`
* `intent`
* `summarization`

Example:
`python suite_benchmark.py --task toxicity --models qwen2.5:7b-instruct llama3.1:8b --sample-size 100`

This writes per-task outputs under `results/suite/`:
* `{task}_summary.csv`
* `{task}_predictions.csv`
* `{task}_visualization.html`

Open the HTML file in a browser to see a per-task visualization of the benchmark metrics for each model.

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
