"""Benchmark settings, read from ezvals.json.

Each entry under "configs" is an EZVals config profile for one model under test.
Pick one with `--config`:

    uv run ezvals run evals/ --config gpt-6-sol --session v3

A profile needs a "model" (an OpenRouter id) and can override any key in
"crunchbench.model_defaults". Context sizes are a shared setting under
"crunchbench".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from crunchbench.data import ROOT

CONFIG_PATH = ROOT / "ezvals.json"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    reasoning_effort: str | None = "high"
    # Output budget including reasoning. None means the model's own maximum, so a
    # cut-off answer means the model couldn't finish, not that we stopped it.
    max_tokens: int | None = None
    temperature: float | None = None
    prompt_cache: bool = True
    # "auto": strict JSON-schema output when OpenRouter lists support for it,
    # otherwise ask for JSON in the prompt. "strict" or "prompt" force one mode.
    structured_output: str = "auto"
    # OpenRouter provider routing preferences, e.g. {"only": ["DeepSeek"]}.
    # https://openrouter.ai/docs/features/provider-routing
    provider: dict | None = None


STRUCTURED_OUTPUT_MODES = ("auto", "strict", "prompt")


@dataclass(frozen=True)
class Settings:
    profiles: dict[str, dict]
    model_defaults: dict
    context_sizes: list[int]

    def model(self, profile: dict | None) -> ModelConfig:
        """Resolve the model under test from the active EZVals config profile."""
        if not profile or "model" not in profile:
            names = ", ".join(self.profiles) or "(none defined)"
            raise ValueError(f"Choose a model profile with --config. Profiles in ezvals.json: {names}")
        merged = {**self.model_defaults, **profile}
        return _model_config(merged["model"], merged)


def _model_config(model_id: str, options: dict) -> ModelConfig:
    mode = options.get("structured_output", "auto")
    if mode not in STRUCTURED_OUTPUT_MODES:
        raise ValueError(f"structured_output must be one of {STRUCTURED_OUTPUT_MODES}, got {mode!r}")
    return ModelConfig(
        id=model_id,
        reasoning_effort=options.get("reasoning_effort"),
        max_tokens=None if options.get("max_tokens") is None else int(options["max_tokens"]),
        temperature=options.get("temperature"),
        prompt_cache=bool(options.get("prompt_cache", False)),
        structured_output=mode,
        provider=options.get("provider"),
    )


@cache
def load_settings(path: Path = CONFIG_PATH) -> Settings:
    raw = json.loads(Path(path).read_text())
    bench = raw.get("crunchbench", {})
    return Settings(
        profiles=raw.get("configs", {}),
        model_defaults=bench.get("model_defaults", {}),
        context_sizes=[int(t) for t in bench.get("context_sizes", [10000, 20000, 30000, 40000, 50000])],
    )
