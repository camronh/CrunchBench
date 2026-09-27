"""CrunchBench as an EZVals suite: 20 questions at each context size.

The model under test comes from the EZVals config profile you pick:

    uv run ezvals run evals/ --config gpt-6-sol --session v3

Profiles live in ezvals.json, and the profile name becomes the run name.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, replace

from ezvals import EvalContext, eval
from langsmith import get_current_run_tree, traceable

from crunchbench.config import ModelConfig, load_settings
from crunchbench.data import context_slice
from crunchbench.dataset import context_hash, load_dataset
from crunchbench.grading import Grade, grade, response_format
from crunchbench.llm import chat, max_output_tokens, use_strict_schema
from crunchbench.prompts import build_messages

SETTINGS = load_settings()


def build_cases() -> list[dict]:
    sizes = set(SETTINGS.context_sizes)
    rows = [c for c in load_dataset()["cases"] if c["tokens"] in sizes]
    missing = sizes - {c["tokens"] for c in rows}
    if missing:
        raise ValueError(f"No dataset cases for context sizes {sorted(missing)}. Run scripts/build_dataset.py.")
    for tokens in sizes:
        want = next(c["context_sha256"] for c in rows if c["tokens"] == tokens)
        if context_hash(context_slice(tokens)[0]) != want:
            raise ValueError(
                f"App data changed since the dataset was built ({tokens} tokens). Run scripts/build_dataset.py."
            )
    # Grouped by context size so consecutive calls share a prompt prefix, which
    # lets providers serve the app data from cache.
    # Within a size, questions with the same answer type run together, because
    # OpenAI and Anthropic put the response schema in the cached prefix.
    rows.sort(key=lambda c: (c["tokens"], c["answer"]["kind"], c["question_id"]))
    return [
        {
            "id": f"{row['question_id']}-{row['tokens'] // 1000}k",
            "input": {"question_id": row["question_id"], "tokens": row["tokens"], "question": row["question"]},
            "reference": row["reference"],
            "labels": [row["difficulty"].lower().replace(" ", "-"), f"{row['tokens'] // 1000}k"],
            "metadata": {
                "difficulty": row["difficulty"],
                "tokens": row["tokens"],
                "num_apps": row["num_apps"],
                "question_id": row["question_id"],
                "answer_kind": row["answer"]["kind"],
                "answer": row["answer"],
            },
        }
        for row in rows
    ]


@traceable(name="CrunchBench", run_type="chain")
async def answer(model: dict, question: str, tokens: int, kind: str, strict: bool) -> dict:
    """Ask the model under test. Traced to LangSmith."""
    model_config = ModelConfig(**model)
    # One OpenRouter session per model and context size keeps those calls on the
    # same provider, so they share its prompt cache.
    provider = dict(model_config.provider or {})
    extra_body: dict = {"session_id": f"crunchbench-{model_config.id}-{tokens}"}
    overrides: dict = {"extra_body": extra_body}
    if strict:
        overrides["response_format"] = response_format(kind)
        # Only route to providers that enforce the schema.
        provider["require_parameters"] = True
    if provider:
        extra_body["provider"] = provider
    completion = await chat(model_config, build_messages(model_config, question, tokens, kind), **overrides)
    return {"completion": completion, "run": get_current_run_tree()}


async def trace_url(run) -> str | None:
    if run is None or os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return None
    try:
        return await asyncio.to_thread(run.get_url)
    except Exception:  # noqa: BLE001 - a missing trace link should never fail the eval.
        return None


@eval(dataset="crunchbench", cases=build_cases(), default_score_key="correctness")
async def crunchbench(ctx: EvalContext):
    model = SETTINGS.model(ctx.config)
    ctx.store(metadata={"model": model.id, "reasoning_effort": model.reasoning_effort})

    strict = await asyncio.to_thread(use_strict_schema, model)
    if model.max_tokens is None:
        model = replace(model, max_tokens=await asyncio.to_thread(max_output_tokens, model, strict))
    ctx.store(metadata={"max_tokens": model.max_tokens})
    expected = ctx.metadata["answer"]
    result = await answer(
        asdict(model),
        ctx.input["question"],
        ctx.input["tokens"],
        expected["kind"],
        strict,
        langsmith_extra={
            "metadata": {
                **{k: v for k, v in ctx.metadata.items() if k != "answer"},
                "ezvals_session": ctx.session_name,
                "ezvals_run": ctx.run_name,
            },
            "tags": [model.id, ctx.input["question_id"], f"{ctx.input['tokens'] // 1000}k"],
        },
    )
    completion = result["completion"]
    usage = completion.usage
    # A model that runs out of output budget mid-reasoning gets no credit, but it
    # is recorded as cut off, not as a formatting mistake.
    cut_off = completion.finish_reason == "length"
    graded = grade(expected, completion.content)
    if cut_off:
        graded = Grade(0.0, "Hit max_tokens before finishing its answer", format_ok=True)

    ctx.store(
        output=completion.content,
        scores={"key": "correctness", "value": graded.score, "passed": graded.passed, "notes": graded.notes},
        metadata={
            "format_ok": graded.format_ok,
            "cut_off": cut_off,
            "structured_output": "strict" if strict else "prompt",
            "cost_usd": completion.cost,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "cache_write_tokens": (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens"),
            "provider": completion.provider,
            "provider_routing": model.provider,
            "model_version": completion.model,
            "finish_reason": completion.finish_reason,
        },
        trace_url=await trace_url(result["run"]),
        # The full prompt lives in the LangSmith trace. Here the app data is
        # summarized so run files stay small.
        messages=[
            {
                "role": "user",
                "content": f"[App Store data: {ctx.metadata['num_apps']} apps, ~{ctx.input['tokens']:,} tokens]\n\n"
                f"{ctx.input['question']}",
            },
            {"role": "assistant", "content": completion.content},
        ],
        trace_data={"reasoning": completion.reasoning},
    )
