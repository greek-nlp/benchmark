# Greek NLP Dataset Mining

## 1) Install requirements

```bash
pip install acl-anthology requests pypdf boto3
```

Optional (for Semantic Scholar rate limits):

```bash
export SEMANTIC_SCHOLAR_API_KEY="your_api_key"
```

## 2) Set AWS credentials

Create `aws.json` in repo root with:

```json
{
  "aws_access_key_id": "...",
  "aws_secret_access_key": "...",
  "aws_region": "..."
}
```

## 3) Build the final CSV (evergreen, by date range)

Run from repo root:

```bash
python dataset_mining/build_gr_dataset_mining.py \
  --min-year 2024 \
  --max-year 2026 \
  --aws-json aws.json \
  --model-id meta.llama3-70b-instruct-v1:0
```

Output (default):

`dataset_mining/gr_dataset_mining_<min>_mid<max>_automatic.csv`

Example:

`dataset_mining/gr_dataset_mining_2024_mid2026_automatic.csv`

## 4) Keep intermediates (optional)

```bash
python dataset_mining/build_gr_dataset_mining.py \
  --min-year 2024 \
  --max-year 2026 \
  --aws-json aws.json \
  --keep-intermediate
```
