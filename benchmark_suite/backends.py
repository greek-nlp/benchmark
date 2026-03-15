from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import error, request


@dataclass
class GenerationResult:
    response: str
    latency_seconds: float


class OllamaBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:11434/api/generate") -> None:
        self.base_url = base_url

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        num_predict: int,
        timeout_seconds: int,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://127.0.0.1:11434. "
                "Start Ollama first, then pull the models you want to benchmark."
            ) from exc

        return GenerationResult(
            response=body["response"].strip(),
            latency_seconds=time.perf_counter() - started,
        )
