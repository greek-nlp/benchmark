# Greek NLP Paper Retrieval (2024-2026)

This folder contains a script to retrieve recent Greek-related NLP papers from:

- ACL Anthology (via `acl-anthology`)
- Semantic Scholar API

## Install

```bash
pip install acl-anthology requests
```

Optional for Semantic Scholar rate limits:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your_api_key"
```

## Run

```bash
python dataset_search_2024_2026/retrieve_greek_nlp_papers.py
```

Custom output:

```bash
python dataset_search_2024_2026/retrieve_greek_nlp_papers.py \
  --output-csv dataset_search_2024_2026/greek_nlp_papers_2024_2026.csv
```

## Output schema

Both sources are saved with the same columns:

- `source`
- `paper_id`
- `title`
- `abstract`
- `year`
- `authors`
- `venue`
- `url`
- `doi`
- `query_match_scope`
- `retrieved_at_utc`
