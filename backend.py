# backend.py
# v1 — Initial implementation. Abstracts Ollama, llama.cpp, vLLM behind
#       a single call_backend() interface. Backend selected via config.BACKEND.
#       Model name translation via config.MODEL_ALIASES.

import json
import requests
from typing import Generator

from config import BACKEND, BACKEND_URLS, MODEL_ALIASES


class BackendConnectionError(Exception):
    """Raised when the active backend cannot be reached (connection or timeout)."""
    def __init__(self, backend: str, url: str, original: Exception):
        self.backend  = backend
        self.url      = url
        self.original = original
        super().__init__(f"[{backend}] Cannot connect to {url}: {original}")


def _resolve_model(model: str) -> str:
    """Translate internal pipeline model name to backend-specific name."""
    return MODEL_ALIASES.get(BACKEND, {}).get(model, model)


def call_backend(
    model: str,
    prompt: str,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    system: str = "",
    stop: list[str] | None = None,
) -> "str | Generator[str, None, None]":
    """
    Send prompt to the configured inference backend.

    model:       internal pipeline model name (resolved via MODEL_ALIASES)
    prompt:      assembled prompt string (skill + context + constraints + user input)
    stream:      if True, return a generator yielding string chunks
    temperature: sampling temperature
    max_tokens:  response length cap; if None, backend default applies
                 (Ollama default is unlimited — do not pass unless needed)
    system:      system context string, injected separately from prompt
    stop:        stop sequences; if None, no stop sequences are sent

    Raises BackendConnectionError on connection or timeout failure.
    HTTP-level errors (4xx/5xx) propagate as requests.exceptions.HTTPError.
    """
    resolved_model = _resolve_model(model)
    base_url = BACKEND_URLS[BACKEND]

    if BACKEND == "ollama":
        return _call_ollama(
            resolved_model, prompt, stream, temperature, max_tokens, system, stop, base_url
        )
    else:
        return _call_openai_compat(
            resolved_model, prompt, stream, temperature, max_tokens, system, stop, base_url
        )


# ─── Ollama backend ───────────────────────────────────────────────────────────

def _call_ollama(
    model: str,
    prompt: str,
    stream: bool,
    temperature: float,
    max_tokens: int | None,
    system: str,
    stop: list[str] | None,
    base_url: str,
) -> "str | Generator[str, None, None]":
    url     = f"{base_url}/api/generate"
    options = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: dict = {
        "model":   model,
        "prompt":  prompt,
        "stream":  stream,
        "options": options,
    }
    if system:
        payload["system"] = system
    if stop is not None:
        payload["stop"] = stop

    if stream:
        return _ollama_stream(url, payload)

    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError as e:
        raise BackendConnectionError("ollama", url, e) from e
    except requests.exceptions.Timeout as e:
        raise BackendConnectionError("ollama", url, e) from e


def _ollama_stream(url: str, payload: dict) -> Generator[str, None, None]:
    """Yield response tokens from Ollama /api/generate with stream=True."""
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError as e:
        raise BackendConnectionError("ollama", url, e) from e
    except requests.exceptions.Timeout as e:
        raise BackendConnectionError("ollama", url, e) from e


# ─── OpenAI-compatible backend (llama.cpp / vLLM) ────────────────────────────

def _call_openai_compat(
    model: str,
    prompt: str,
    stream: bool,
    temperature: float,
    max_tokens: int | None,
    system: str,
    stop: list[str] | None,
    base_url: str,
) -> "str | Generator[str, None, None]":
    url      = f"{base_url}/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "stream":      stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop is not None:
        payload["stop"] = stop

    if stream:
        return _openai_stream(url, payload)

    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError as e:
        raise BackendConnectionError(BACKEND, url, e) from e
    except requests.exceptions.Timeout as e:
        raise BackendConnectionError(BACKEND, url, e) from e


def _openai_stream(url: str, payload: dict) -> Generator[str, None, None]:
    """Yield content tokens from an OpenAI-compatible SSE stream."""
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                if not text.startswith("data: "):
                    continue
                data = text[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if content:
                    yield content
    except requests.exceptions.ConnectionError as e:
        raise BackendConnectionError(BACKEND, url, e) from e
    except requests.exceptions.Timeout as e:
        raise BackendConnectionError(BACKEND, url, e) from e
