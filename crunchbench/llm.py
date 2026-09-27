"""OpenRouter chat calls, traced to LangSmith when LANGSMITH_TRACING is on."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
import weakref
from dataclasses import dataclass, field
from functools import cache

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from crunchbench.config import ModelConfig
from crunchbench.data import ROOT

load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# One client per event loop. `ezvals serve` starts a new loop for every run, and
# an async client's connections can't be reused once their loop has closed.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI] = weakref.WeakKeyDictionary()


def client() -> AsyncOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env (see .env.example)")
    loop = asyncio.get_running_loop()
    if loop not in _clients:
        raw = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            max_retries=3,
            # Long-context reasoning can take several minutes on slower providers.
            timeout=1800,
            default_headers={"HTTP-Referer": "https://github.com/camronh/CrunchBench", "X-Title": "CrunchBench"},
        )
        _clients[loop] = wrap_openai(raw)
    return _clients[loop]


@dataclass
class Completion:
    content: str
    reasoning: str | None
    model: str
    provider: str | None
    finish_reason: str | None
    usage: dict = field(default_factory=dict)

    @property
    def cost(self) -> float | None:
        return self.usage.get("cost")


@cache
def supports_structured_outputs(model_id: str) -> bool:
    """Whether OpenRouter lists strict structured-output support for this model."""
    with urllib.request.urlopen(f"{OPENROUTER_BASE_URL}/models", timeout=30) as response:
        models = json.load(response)["data"]
    for m in models:
        if m["id"] == model_id:
            return "structured_outputs" in (m.get("supported_parameters") or [])
    raise ValueError(f"Model {model_id!r} is not on OpenRouter")


@cache
def _endpoints(model_id: str) -> list[dict]:
    with urllib.request.urlopen(f"{OPENROUTER_BASE_URL}/models/{model_id}/endpoints", timeout=30) as response:
        return json.load(response)["data"]["endpoints"]


def max_output_tokens(model: ModelConfig, strict: bool) -> int:
    """The largest output budget every provider this request could reach will accept.

    Considers only the pinned providers, if any, and only providers that support
    strict schemas when the request uses one.
    """
    only = set((model.provider or {}).get("only") or [])
    candidates = [
        e
        for e in _endpoints(model.id)
        if (not only or e["provider_name"] in only)
        and (not strict or "structured_outputs" in (e.get("supported_parameters") or []))
        and e.get("max_completion_tokens")
    ]
    if not candidates:
        raise ValueError(f"No OpenRouter endpoint for {model.id} reports a max output size for this routing")
    return min(e["max_completion_tokens"] for e in candidates)


def use_strict_schema(model: ModelConfig) -> bool:
    if model.structured_output == "auto":
        return supports_structured_outputs(model.id)
    return model.structured_output == "strict"


def request_params(model: ModelConfig) -> dict:
    """Chat-completion parameters for a configured model."""
    if model.max_tokens is None:
        raise ValueError("Resolve max_tokens (see max_output_tokens) before calling the model")
    params: dict = {"model": model.id, "max_tokens": model.max_tokens}
    if model.temperature is not None:
        params["temperature"] = model.temperature
    extra: dict = {"usage": {"include": True}}
    if model.reasoning_effort:
        extra["reasoning"] = {"effort": model.reasoning_effort}
    params["extra_body"] = extra
    return params


async def chat(model: ModelConfig, messages: list[dict], **overrides) -> Completion:
    params = request_params(model)
    extra = {**params.pop("extra_body"), **overrides.pop("extra_body", {})}
    response = await client().chat.completions.create(messages=messages, extra_body=extra, **params, **overrides)
    if not response.choices:
        raise RuntimeError(f"{model.id} returned no choices: {response.model_dump()}")
    choice = response.choices[0]
    data = response.model_dump()
    raw_choice = data["choices"][0]
    # OpenRouter can report a provider failure inside a 200 response. Raise so it
    # is recorded as an error and not scored as the model's answer.
    if choice.finish_reason == "error" or raw_choice.get("error"):
        raise RuntimeError(f"{model.id} provider error: {raw_choice.get('error') or raw_choice}")
    message = raw_choice["message"]
    return Completion(
        content=(choice.message.content or "").strip(),
        reasoning=message.get("reasoning"),
        model=response.model,
        provider=data.get("provider"),
        finish_reason=choice.finish_reason,
        usage=data.get("usage") or {},
    )
