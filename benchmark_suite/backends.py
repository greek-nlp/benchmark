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
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/api/generate",
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

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
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=timeout_seconds) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                last_error = exc
                if exc.code not in {500, 502, 503, 504} or attempt == self.max_retries:
                    raise RuntimeError(
                        f"Ollama request failed for model '{model}' with HTTP {exc.code}. "
                        "The server responded but could not complete generation."
                    ) from exc
            except error.URLError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Could not reach Ollama at http://127.0.0.1:11434. "
                        "Start Ollama first, then pull the models you want to benchmark."
                    ) from exc

            time.sleep(self.retry_backoff_seconds * (attempt + 1))
        else:
            raise RuntimeError("Ollama generation failed after retries.") from last_error

        return GenerationResult(
            response=body["response"].strip(),
            latency_seconds=time.perf_counter() - started,
        )
