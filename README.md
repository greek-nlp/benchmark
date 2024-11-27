# The Greek NLP Benchmark

While large language models (LLMs) have significantly advanced Natural Language Processing (NLP), progress for resource-scarce languages lags behind well-supported ones. Consequently, less-supported languages inherit often non-suited biases and assumptions from well-resourced languages, a gap that is exacerbated by the scarcity of open-access, high-quality language resources. Our work addresses this issue for (Modern) Greek, by collecting and unifying publicly available datasets (created after 2012) into a comprehensive resource. The selected datasets are classified by availability, licensing, and task-specific usage, adhering to the FAIR Data Principles to mitigate biases and to avoid data contamination. We used an open-source (Llama-70b) and a closed-source (GPT-4o mini) LLM to benchmark seven well-known NLP tasks: Toxicity Detection, Grammatical Error Correction, Machine Translation (MT), Summarization, Intent Classification, Named Entity Recognition (NER), Part-Of-Speech (POS) Tagging. Our results show that no model outperforms the other across tasks. Furthermore, we introduce the first benchmark of Text Clustering in Greek, showing that MT (from Greek to English) and Summarization (reducing long texts) along with dense embeddings outperforms TF-IDF when representing long legal documents. Also, we present the first benchmark of Authorship Attribution on publicly available and properly licensed Greek data, revealing the inclusion of authors' works in the LLMs' training data. Repeated at scale, this attribution benchmark can potentially reveal improper data usage. Last, we also assess publicly available, accessible, properly licensed raw data, which could be useful for pre-training purposes.

## Guidelines
* Explore the Greek datasets in [data.csv](data.csv). 
* Run [the data notebook](nlp_gr_access_data.ipynb) to download the data.
* Run the experiments for [the task benchmarks](nlp_gr_experiments.ipynb) for the open-source Llama model. 

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
