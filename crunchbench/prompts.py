"""The prompt each model under test receives."""

from __future__ import annotations

import json

from crunchbench.config import ModelConfig
from crunchbench.data import context_slice
from crunchbench.grading import SCHEMAS
from crunchbench.questions import RULES

INSTRUCTIONS = """\
Answer the question below using the <APP STORE DATA> above.

<RULES>
{rules}
</RULES>

<QUESTION>
{question}
</QUESTION>

Respond with only a JSON object that matches this schema. When the question asks for an order, list the items in that order.

{schema}"""


def build_messages(model: ModelConfig, question: str, tokens: int, kind: str) -> list[dict]:
    """One user message: the app data first (so it can be cached), then the question."""
    data, _ = context_slice(tokens)
    data_block = {"type": "text", "text": f"<APP STORE DATA>\n{data}\n</APP STORE DATA>"}
    if model.prompt_cache:
        # Mark the end of the shared app data as a cache breakpoint, so questions
        # at the same context size reuse it. Anthropic only caches at explicit
        # breakpoints. OpenAI GPT-5.6+ otherwise only caches at the end of the
        # whole message, which includes the question and never repeats.
        # DeepSeek and most others cache shared prefixes automatically.
        if model.id.startswith("anthropic/"):
            data_block["cache_control"] = {"type": "ephemeral"}
        elif model.id.startswith("openai/"):
            data_block["prompt_cache_breakpoint"] = {"mode": "explicit"}
    text = INSTRUCTIONS.format(rules=RULES, question=question, schema=json.dumps(SCHEMAS[kind], indent=2))
    return [{"role": "user", "content": [data_block, {"type": "text", "text": text}]}]
