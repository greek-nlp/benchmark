# Greek NLP Benchmark Leaderboard Space

This folder contains a minimal Hugging Face Space app for the benchmark leaderboard.

## Files

- `app.py`: Gradio app
- `requirements.txt`: Space dependencies

## Expected Inputs

The app reads:

- `leaderboard_results.jsonl`
- `leaderboard/model_registry.yaml`

By default it points to the latest local exported run in this repository. In a Space, set:

- `LEADERBOARD_RESULTS_PATH`
- `LEADERBOARD_MODEL_REGISTRY`

to the desired mounted or copied files.

## Local Run

```bash
pip install -r leaderboard_space/requirements.txt
python leaderboard_space/app.py
```

## Suggested Space Layout

```text
app.py
requirements.txt
data/
  leaderboard_results.jsonl
  model_registry.yaml
```

Then set:

- `LEADERBOARD_RESULTS_PATH=data/leaderboard_results.jsonl`
- `LEADERBOARD_MODEL_REGISTRY=data/model_registry.yaml`
