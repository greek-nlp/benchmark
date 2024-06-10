# The GenA of Greek NLP: A Systematic Survey

__Αbstract__: Natural Language Processing (NLP) focuses on enabling machines to understand and generate human language. While deep learning and large language models have significantly advanced NLP, progress for resource-scarce languages lags behind well-supported ones; thus, less-supported languages inherit often non-suited biases and assumptions from well-resourced languages. This gap is exacerbated by the scarcity of open-access, high-quality language resources. This work addresses this issue for Modern Greek, by collecting and unifying into a comprehensive resource publicly available datasets created after 2012. These datasets are classified by availability, licensing, and usage, adhering to FAIR Data Principles to mitigate biases and data contamination. We used GPT-3.5-turbo and 4o to undertake a benchmark on five NLP tasks, and we merged the rest into a dataset of 1.7 billion characters, potentially useful for pre-training purposes. In one of the tasks, text clustering, we also set the first Modern Greek benchmark.

## Guidelines
Run the Colaboratory notebook in this repository to download the data, to use character-level statistical language modelling for exploratory analysis, and to undertake a text clustering benchmark. LLM predictions for downstream tasks besides clustering, along with the code to produce them, are also included. 


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
