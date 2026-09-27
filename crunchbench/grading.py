"""Response schemas and deterministic grading against the ground truth.

Scoring:
- 1.0: exactly the right items (in the right order when order matters), or the right number.
- 0.5: the ground truth has several items and the answer has at least one of them,
  but misses some, adds others or has them out of order.
- 0.0: none of the right items, a wrong count or number, or output that
  doesn't match the schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from crunchbench.questions import APPS, CATEGORIES, COUNT, NUMBER


def _object(prop: str, schema: dict) -> dict:
    return {"type": "object", "properties": {prop: schema}, "required": [prop], "additionalProperties": False}


APP_ITEM = {
    "type": "object",
    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
    "required": ["id", "name"],
    "additionalProperties": False,
}

SCHEMAS = {
    APPS: _object("apps", {"type": "array", "items": APP_ITEM}),
    CATEGORIES: _object("categories", {"type": "array", "items": {"type": "string"}}),
    COUNT: _object("count", {"type": "integer"}),
    NUMBER: _object("value", {"type": "number"}),
}

FIELD = {APPS: "apps", CATEGORIES: "categories", COUNT: "count", NUMBER: "value"}


def response_format(kind: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": f"crunchbench_{kind}", "strict": True, "schema": SCHEMAS[kind]},
    }


class FormatError(ValueError):
    """The model's output could not be read as an answer of the expected kind."""


def parse_json(text: str) -> dict:
    """Read a JSON object from model output, tolerating code fences or surrounding prose."""
    text = (text or "").strip()
    candidates = [text]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise FormatError("no JSON object found in the output")


def read_answer(kind: str, text: str) -> list:
    """Extract the answer values: app IDs, category names, or one number."""
    data = parse_json(text)
    field = FIELD[kind]
    if field not in data:
        raise FormatError(f'missing "{field}" field')
    value = data[field]

    if kind == APPS:
        if not isinstance(value, list):
            raise FormatError('"apps" is not a list')
        ids = []
        for item in value:
            raw = item.get("id") if isinstance(item, dict) else None
            if isinstance(raw, bool) or raw is None:
                raise FormatError(f"app entry without an integer id: {item!r}")
            try:
                ids.append(int(str(raw).strip()))
            except ValueError as e:
                raise FormatError(f"app id is not an integer: {raw!r}") from e
        return ids
    if kind == CATEGORIES:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise FormatError('"categories" is not a list of strings')
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FormatError(f'"{field}" is not a number: {value!r}')
    if kind == COUNT and float(value) != int(value):
        raise FormatError(f"count is not a whole number: {value!r}")
    return [value]


@dataclass
class Grade:
    score: float
    notes: str
    format_ok: bool = True

    @property
    def passed(self) -> bool:
        return self.score == 1.0


def _norm_category(name: str) -> str:
    return " ".join(name.split()).casefold()


def grade(answer: dict, output: str) -> Grade:
    """Grade raw model output against a published answer (see Answer.to_json)."""
    kind = answer["kind"]
    try:
        got = read_answer(kind, output)
    except FormatError as e:
        return Grade(0.0, f"Invalid output: {e}", format_ok=False)

    if kind in (COUNT, NUMBER):
        want = answer["value"]
        value = got[0]
        if kind == NUMBER:
            # Any correct rounding of the exact value, or the exact value itself.
            ok = abs(float(value) - want) <= 0.5 * 10 ** -answer["decimals"] + 1e-9
        else:
            ok = int(value) == want
        shown = f"{want:.{answer['decimals']}f}" if kind == NUMBER else want
        return Grade(1.0 if ok else 0.0, f"Expected {shown}, got {value}")

    if kind == APPS:
        want = [item["id"] for item in answer["items"]]
        names = {item["id"]: item["name"] for item in answer["items"]}
    else:
        want = [_norm_category(c) for c in answer["items"]]
        got = [_norm_category(c) for c in got]
        names = {_norm_category(c): c for c in answer["items"]}

    label = lambda x: names.get(x, str(x))
    if not want:
        if not got:
            return Grade(1.0, "Correctly returned an empty list")
        return Grade(0.0, f"Expected an empty list, got {len(got)} item(s)")

    want_set, got_set = set(want), set(got)
    missing = [label(x) for x in want if x not in got_set]
    extra = [str(x) for x in got if x not in want_set]
    duplicates = len(got) != len(got_set)
    order_ok = not answer["ordered"] or got == want

    if not missing and not extra and not duplicates and order_ok:
        return Grade(1.0, "Exact match")

    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"extra {extra}")
    if duplicates:
        problems.append("duplicate items")
    if not missing and not extra and not order_ok:
        problems.append("wrong order")
    overlap = want_set & got_set
    score = 0.5 if overlap and len(want) > 1 else 0.0
    return Grade(score, "; ".join(problems))
