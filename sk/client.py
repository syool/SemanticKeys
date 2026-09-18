"""
client.py — Multi-backend LLM client module.

Exposes a uniform interface (LLMResponse + .call()) across:
  - AnthropicClient          : Claude via Messages API
  - OpenAICompatibleClient   : any OpenAI /v1/chat/completions endpoint
                               (vLLM, Ollama, LM Studio, OpenAI, Together,
                                Fireworks, Groq, etc.)
  - GoogleClient             : Gemini via generateContent API
  - SimulatedClient          : deterministic heuristic fallback, no network

The experiment runner depends only on LLMResponse and .call(system, user,
max_tokens). This lets main model and judge be any combination of backends.

Typical wiring for the Hybrid setup on L40S:

    from client import OpenAICompatibleClient, AnthropicClient
    main_client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-27B-A3B-Instruct",
    )
    judge_client = AnthropicClient(model="claude-sonnet-4-5")

All clients record input/output token counts authoritatively from the API
response where available; they fall back to heuristic count_tokens only
when the backend doesn't return usage info.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def count_tokens(text: str) -> int:
    """
    Coarse token estimate used only as fallback when backend doesn't report
    usage. Should not be used for paper numbers — real backends report exact
    counts in the usage field of responses.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(words, int(chars / 4))


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    finish_reason: str = ""
    # Populated when a backend returns chain-of-thought separately from the
    # answer. Non-empty means reasoning was generated and billed to
    # completion_tokens, which invalidates Experiment 1's token comparison.
    reasoning: str = ""
    raw: dict | None = None


class _HTTPError(Exception):
    pass


def _http_post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    """
    Minimal JSON POST returning parsed JSON. Raises _HTTPError on non-2xx
    or transport failures. Dependency-free to keep this module lightweight.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise _HTTPError(f"HTTP {e.code}: {body[:1000]}")
    except urllib.error.URLError as e:
        raise _HTTPError(f"URL error: {e.reason}")


# =============================================================================
# Anthropic
# =============================================================================


class AnthropicClient:
    """Client for Anthropic Messages API. Used as judge in the Hybrid setup."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        api_url: str = "https://api.anthropic.com/v1/messages",
        anthropic_version: str = "2023-06-01",
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_version = anthropic_version

    def call(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        start = time.time()
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        try:
            data = _http_post_json(self.api_url, headers, payload)
        except _HTTPError as e:
            latency = (time.time() - start) * 1000
            return LLMResponse(
                content=f"ERROR: {e}",
                input_tokens=count_tokens(system + user),
                output_tokens=0,
                total_tokens=count_tokens(system + user),
                latency_ms=latency,
                model=self.model,
                finish_reason="error",
            )

        latency = (time.time() - start) * 1000
        content_blocks = data.get("content", [])
        content = ""
        for blk in content_blocks:
            if blk.get("type") == "text":
                content = blk.get("text", "")
                break
        usage = data.get("usage", {})
        in_tok = usage.get("input_tokens", count_tokens(system + user))
        out_tok = usage.get("output_tokens", count_tokens(content))
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            latency_ms=latency,
            model=self.model,
            finish_reason=data.get("stop_reason", ""),
            raw=data,
        )


# =============================================================================
# OpenAI-compatible (vLLM, Ollama, OpenAI, Together, Fireworks, Groq, ...)
# =============================================================================


class OpenAICompatibleClient:
    """
    Client for any OpenAI /v1/chat/completions-compatible endpoint.

    Works with:
      - vLLM (`vllm serve ... --host 0.0.0.0 --port 8000`) — L40S setup
      - Ollama (endpoint http://localhost:11434/v1)
      - LM Studio (built-in OpenAI server)
      - OpenAI (https://api.openai.com/v1)
      - Together AI, Fireworks AI, Groq, DeepInfra, etc.

    Usage for vLLM (no API key needed locally):

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-27B-A3B-Instruct",
            api_key="not-needed",
        )
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self.disable_thinking = disable_thinking

    def call(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        start = time.time()
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }

        # Reasoning-capable models (Qwen 3.x, Gemma 4, and others) emit their
        # chain of thought as ordinary output unless told not to. That is fatal
        # for this experiment specifically: Experiment 1 measures output tokens
        # as a proxy for protocol verbosity, so reasoning tokens would be
        # counted as protocol cost and would swamp the effect being measured.
        # Worse, long reasoning pushes the actual answer past max_tokens, which
        # silently zeroes every accuracy metric downstream.
        #
        # chat_template_kwargs is passed through to the model's chat template.
        # Models without an enable_thinking switch ignore it harmlessly.
        if self.disable_thinking:
            ctk = dict(payload.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = ctk

        # extra_body last so callers can override anything above, including
        # the thinking setting if a model needs a different key.
        payload.update(self.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        try:
            data = _http_post_json(
                f"{self.base_url}/chat/completions", headers, payload
            )
        except _HTTPError as e:
            latency = (time.time() - start) * 1000
            return LLMResponse(
                content=f"ERROR: {e}",
                input_tokens=count_tokens(system + user),
                output_tokens=0,
                total_tokens=count_tokens(system + user),
                latency_ms=latency,
                model=self.model,
                finish_reason="error",
            )

        latency = (time.time() - start) * 1000
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(
                content="ERROR: empty choices",
                input_tokens=0, output_tokens=0, total_tokens=0,
                latency_ms=latency, model=self.model, finish_reason="error",
                raw=data,
            )
        choice = choices[0]
        message = choice.get("message", {}) or {}
        content = message.get("content", "") or ""
        finish = choice.get("finish_reason", "")

        # Some backends return reasoning separately rather than inline. When
        # that happens the tokens still count toward completion_tokens, so
        # Experiment 1 would attribute them to protocol verbosity. Record it
        # so sanity.py can flag the run rather than reporting inflated
        # compression figures as if they were real.
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""

        usage = data.get("usage", {}) or {}
        in_tok = usage.get("prompt_tokens", count_tokens(system + user))
        out_tok = usage.get("completion_tokens", count_tokens(content))
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            latency_ms=latency,
            model=self.model,
            finish_reason=finish,
            reasoning=reasoning,
            raw=data,
        )


# =============================================================================
# Google (Gemini)
# =============================================================================


class GoogleClient:
    """Client for Google Gemini via generateContent API."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def call(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        start = time.time()
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.0,
            },
        }
        headers = {"Content-Type": "application/json"}
        try:
            data = _http_post_json(url, headers, payload)
        except _HTTPError as e:
            latency = (time.time() - start) * 1000
            return LLMResponse(
                content=f"ERROR: {e}",
                input_tokens=count_tokens(system + user),
                output_tokens=0,
                total_tokens=count_tokens(system + user),
                latency_ms=latency,
                model=self.model,
                finish_reason="error",
            )

        latency = (time.time() - start) * 1000
        candidates = data.get("candidates", [])
        if not candidates:
            return LLMResponse(
                content="ERROR: no candidates",
                input_tokens=0, output_tokens=0, total_tokens=0,
                latency_ms=latency, model=self.model, finish_reason="error",
                raw=data,
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)
        finish = candidates[0].get("finishReason", "")
        usage = data.get("usageMetadata", {}) or {}
        in_tok = usage.get("promptTokenCount", count_tokens(system + user))
        out_tok = usage.get("candidatesTokenCount", count_tokens(content))
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            latency_ms=latency,
            model=self.model,
            finish_reason=finish,
            raw=data,
        )


# =============================================================================
# Simulated (no network, deterministic, for pipeline validation only)
# =============================================================================


class SimulatedClient:
    """
    Deterministic heuristic client for pipeline validation without network.
    Do NOT use for paper numbers.
    """

    def __init__(self, model: str = "simulated-baseline"):
        self.model = model

    def call(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        start = time.time()
        if "Semantic Keys" in system or "semantic_keys" in system.lower():
            content = self._simulate_sk_parse(user)
        elif '"content"' in system and "JSON" in system:
            content = self._simulate_json_free(user)
        else:
            content = self._simulate_nl_response(user)

        latency = (time.time() - start) * 1000
        in_tok = count_tokens(system + user)
        out_tok = count_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            latency_ms=latency,
            model=self.model,
            finish_reason="stop",
        )

    def _simulate_sk_parse(self, text: str) -> str:
        result = {}
        sentences = text.split(".")
        if sentences:
            words = sentences[0].strip().split()
            if len(words) > 2:
                result["topic"] = " ".join(words[:4])
        for sent in sentences:
            low = sent.lower().strip()
            if any(w in low for w in ["who ", "person", "agent"]):
                result["who"] = sent.strip()[:80]
            if any(w in low for w in ["when", "date", "year"]):
                result["when"] = sent.strip()[:80]
            if any(w in low for w in ["where", "location", "city"]):
                result["where"] = sent.strip()[:80]
        result["what"] = text[:100].strip()
        result["evidence"] = f"Source text ({len(text)} chars)"
        return json.dumps(result, ensure_ascii=False)

    def _simulate_json_free(self, text: str) -> str:
        return json.dumps({
            "content": f"Analysis of: {text[:150]}",
            "answer": "simulated answer",
        })

    def _simulate_nl_response(self, text: str) -> str:
        if "?" in text:
            return f"Based on the information, the answer involves analyzing the question. {text[:200]}"
        return f"I have reviewed the information. {text[:300]}"


# =============================================================================
# Factory helper
# =============================================================================


def make_client(spec: dict) -> Any:
    """
    Build a client from a config dict. Makes the pilot script configurable
    without touching code — just edit the JSON/dict.

    Example specs:
        {"backend": "vllm", "base_url": "http://localhost:8000/v1",
         "model": "Qwen/Qwen3-27B-A3B-Instruct"}
        {"backend": "anthropic", "model": "claude-sonnet-4-5"}
        {"backend": "openai", "model": "gpt-4o"}
        {"backend": "google", "model": "gemini-2.5-pro"}
        {"backend": "simulated"}
    """
    backend = spec["backend"]
    if backend == "anthropic":
        return AnthropicClient(model=spec.get("model", "claude-sonnet-4-5"))
    if backend in ("vllm", "ollama", "openai-compatible"):
        return OpenAICompatibleClient(
            base_url=spec["base_url"],
            model=spec["model"],
            api_key=spec.get("api_key"),
            extra_body=spec.get("extra_body"),
        )
    if backend == "openai":
        return OpenAICompatibleClient(
            base_url=spec.get("base_url", "https://api.openai.com/v1"),
            model=spec["model"],
            api_key=spec.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        )
    if backend == "google":
        return GoogleClient(
            model=spec.get("model", "gemini-2.5-pro"),
            api_key=spec.get("api_key"),
        )
    if backend == "simulated":
        return SimulatedClient(model=spec.get("model", "simulated-baseline"))
    raise ValueError(f"Unknown backend: {backend!r}")