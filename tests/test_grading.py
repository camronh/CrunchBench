"""Unit tests for the deterministic grader."""

from __future__ import annotations

import json

import pytest

from crunchbench.grading import SCHEMAS, grade, parse_json

APPS3 = {
    "kind": "apps",
    "ordered": True,
    "items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}],
}
APPS_UNORDERED = {**APPS3, "ordered": False}
ONE_APP = {"kind": "apps", "ordered": False, "items": [{"id": 1, "name": "A"}]}
NO_APPS = {"kind": "apps", "ordered": False, "items": []}
CATS = {"kind": "categories", "ordered": True, "items": ["Games", "Photo & Video"]}
COUNT = {"kind": "count", "ordered": False, "value": 23}
PRICE = {"kind": "number", "ordered": False, "value": 2.2857142857, "decimals": 2}
EDGE = {"kind": "number", "ordered": False, "value": 4.115, "decimals": 2}


def apps(*ids):
    return json.dumps({"apps": [{"id": i, "name": f"app {i}"} for i in ids]})


@pytest.mark.parametrize(
    ("answer", "output", "score"),
    [
        (APPS3, apps(1, 2, 3), 1.0),
        (APPS3, apps(2, 1, 3), 0.5),  # wrong order
        (APPS3, apps(1, 2), 0.5),  # missing one
        (APPS3, apps(1, 2, 3, 4), 0.5),  # extra one
        (APPS3, apps(1, 1, 2, 3), 0.5),  # duplicate
        (APPS3, apps(7, 8, 9), 0.0),
        (APPS3, apps(), 0.0),
        (APPS_UNORDERED, apps(3, 1, 2), 1.0),
        (ONE_APP, apps(1), 1.0),
        (ONE_APP, apps(1, 2), 0.0),  # single answer with an extra item is simply wrong
        (ONE_APP, apps(2), 0.0),
        (NO_APPS, apps(), 1.0),
        (NO_APPS, apps(5), 0.0),
        (CATS, '{"categories": ["games", "Photo  &  Video"]}', 1.0),  # case and spacing don't matter
        (CATS, '{"categories": ["Photo & Video", "Games"]}', 0.5),
        (CATS, '{"categories": ["Games"]}', 0.5),
        (COUNT, '{"count": 23}', 1.0),
        (COUNT, '{"count": 23.0}', 1.0),
        (COUNT, '{"count": 22}', 0.0),
        (PRICE, '{"value": 2.29}', 1.0),
        (PRICE, '{"value": 2.2861}', 1.0),  # rounds to the requested precision
        (PRICE, '{"value": 2.3}', 0.0),
        (PRICE, '{"value": 2.28}', 0.0),
        (EDGE, '{"value": 4.11}', 1.0),  # exact value on a .5 boundary: both roundings count
        (EDGE, '{"value": 4.12}', 1.0),
        (EDGE, '{"value": 4.13}', 0.0),
    ],
)
def test_scores(answer, output, score):
    assert grade(answer, output).score == score


def test_string_ids_are_accepted():
    assert grade(APPS3, '{"apps": [{"id": "1", "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}]}').passed


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Facebook, Pinterest",
        '{"categories": ["Games"]}',  # wrong field for an apps answer
        '{"apps": [{"name": "A"}]}',  # no id
        '{"apps": "A, B"}',
    ],
)
def test_format_failures(output):
    result = grade(APPS3, output)
    assert result.score == 0.0
    assert not result.format_ok


def test_count_rejects_fractions_and_bools():
    assert not grade(COUNT, '{"count": 22.5}').format_ok
    assert not grade(COUNT, '{"count": true}').format_ok


@pytest.mark.parametrize(
    "text",
    [
        '{"count": 3}',
        '```json\n{"count": 3}\n```',
        'Here is the answer:\n{"count": 3}\nDone.',
    ],
)
def test_parse_json_tolerates_wrappers(text):
    assert parse_json(text) == {"count": 3}


def test_schemas_are_strict_compatible():
    def check(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for sub in schema["properties"].values():
                check(sub)
        if schema.get("type") == "array":
            check(schema["items"])

    for schema in SCHEMAS.values():
        check(schema)
