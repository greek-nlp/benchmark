from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class TaskSpec:
    name: str
    system_prompt: str
    load_dataset: Callable[..., pd.DataFrame]
    build_prompt: Callable[[dict[str, object]], str]
    evaluate: Callable[[pd.DataFrame], pd.DataFrame]
    normalize_prediction: Callable[[str, dict[str, object]], object] = lambda text, _example: " ".join(text.split())
