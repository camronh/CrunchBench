"""Independent re-derivation of every ground truth.

These tests do not reuse pandas or anything in `crunchbench.questions`. They parse
the exact context text a model receives, recompute each answer in plain Python,
and compare with the published dataset. Rankings also have to be strictly
separated, so a tie can never decide an answer.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from crunchbench.data import context_slice

DATASET = json.loads((Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json").read_text())
SIZES = sorted({c["tokens"] for c in DATASET["cases"]})

# Every field but the description is one line. Descriptions can contain field-like
# text (one says "Description:" again), so only the description may span lines.
FIELD = re.compile(
    r"App Name: (?P<name>[^\n]*)\nID: (?P<id>\d+)\nSize: (?P<size>[\d.]+) MB\nPrice: (?P<price>[\d.]+) USD\n"
    r"Rating Count: (?P<count>\d+)\nAverage User Rating: (?P<rating>[\d.]+)\nVersion: (?P<ver>[^\n]*)\n"
    r"Category: (?P<cat>[^\n]*)\nDescription: (?P<desc>.*)",
    re.DOTALL,
)


def parse(text: str) -> list[dict]:
    apps = []
    for block in text.split("\n\n==============\n\n"):
        m = FIELD.fullmatch(block)
        assert m, f"could not parse block: {block[:120]!r}"
        apps.append(
            {
                "name": m["name"],
                "id": int(m["id"]),
                "size": Fraction(m["size"]),
                "price": Fraction(m["price"]),
                "count": int(m["count"]),
                "rating": Fraction(m["rating"]),
                "cat": m["cat"],
                "desc": m["desc"],
            }
        )
    return apps


def top(rows, key, k):
    """Top k by key, requiring the key to strictly separate positions 1..k+1."""
    rows = sorted(rows, key=key)
    keys = [key(r) for r in rows[: k + 1]]
    assert len(set(keys)) == len(keys), f"tie in ranking: {keys}"
    return rows[:k]


def best_group(values: dict, k=1):
    ranked = sorted(values.items(), key=lambda kv: -kv[1])
    vals = [v for _, v in ranked[: k + 1]]
    assert len(set(vals)) == len(vals), f"tie between groups: {ranked[: k + 1]}"
    return [g for g, _ in ranked[:k]]


def word(w):
    return lambda a: (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", a["desc"], re.IGNORECASE) is not None
    )


def by_cat(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["cat"]].append(r)
    return out


def names(rows):
    """Apps are graded by ID."""
    return [r["id"] for r in rows]


def mean(xs):
    return sum(xs, Fraction(0)) / len(xs)


def expected(qid: str, apps: list[dict]) -> list[str]:
    free = [a for a in apps if a["price"] == 0]
    paid = [a for a in apps if a["price"] > 0]
    games = [a for a in apps if a["cat"] == "Games"]

    if qid == "E1":
        return names(top(apps, lambda a: -a["count"], 3))
    if qid == "E2":
        return names(top(apps, lambda a: -a["size"], 1))
    if qid == "E3":
        return names(top(games, lambda a: -a["count"], 5))
    if qid == "E4":
        return [len(games)]
    if qid == "E5":
        hi = max(a["price"] for a in apps)
        return sorted(a["id"] for a in apps if a["price"] == hi)
    if qid == "M1":
        return names(top(free, lambda a: (-a["rating"], -a["count"]), 3))
    if qid == "M2":
        return names(top(paid, lambda a: (a["rating"], a["count"]), 5))
    if qid == "M3":
        return [sum(1 for a in free if a["rating"] >= Fraction(9, 2))]
    if qid == "M4":
        return best_group({c: sum(a["count"] for a in rs) for c, rs in by_cat(paid).items()})
    if qid == "M5":
        return names(top([a for a in apps if word("offline")(a)], lambda a: -a["count"], 3))
    if qid == "H1":
        return names(top(paid, lambda a: -Fraction(a["count"]) / a["price"], 3))
    if qid == "H2":
        groups = {c: mean([a["rating"] for a in rs]) for c, rs in by_cat(apps).items() if len(rs) >= 6}
        return best_group(groups, k=min(3, len(groups)))
    if qid == "H3":
        return [mean([a["price"] for a in games if a["price"] > 0])]
    if qid == "H4":
        hits = [a for a in games if word("multiplayer")(a) or word("achievements")(a)]
        return names(top(hits, lambda a: (-a["rating"], -a["count"]), 3))
    if qid == "H5":
        hits = [a for a in apps if a["size"] > 200 and a["rating"] >= Fraction(9, 2)]
        return names(top(hits, lambda a: -a["size"], 10))
    if qid == "V1":
        ratios = {}
        for c, rs in by_cat(paid).items():
            if len(rs) >= 3:
                ratios[c] = mean([a["rating"] for a in rs]) / mean([a["price"] for a in rs])
        return best_group(ratios)
    if qid == "V2":
        sds = {c: statistics.pstdev([float(a["rating"]) for a in rs]) for c, rs in by_cat(apps).items() if len(rs) >= 5}
        ranked = sorted(sds.values(), reverse=True)
        assert len(ranked) == 1 or ranked[0] - ranked[1] > 1e-4, "standard deviations too close"
        return [max(sds, key=sds.get)]
    if qid == "V3":
        good = by_cat([a for a in apps if a["rating"] >= Fraction(9, 2)])
        rows = [{"cat": c, "n": len(rs), "sum": sum(a["count"] for a in rs)} for c, rs in good.items()]
        return [r["cat"] for r in top(rows, lambda r: (-r["n"], -r["sum"]), 3)]
    if qid == "V4":
        big = by_cat([a for a in apps if a["size"] > 100])
        return best_group({c: mean([a["count"] for a in rs]) for c, rs in big.items() if len(rs) >= 3})
    if qid == "V5":
        cat_mean = {c: mean([a["rating"] for a in rs]) for c, rs in by_cat(apps).items()}
        above = [a for a in apps if a["rating"] > cat_mean[a["cat"]]]
        return names(top(above, lambda a: -a["count"], 3))
    raise KeyError(qid)


def published(answer: dict) -> list:
    """The published answer in the same shape `expected` returns."""
    if answer["kind"] == "apps":
        ids = [item["id"] for item in answer["items"]]
        return ids if answer["ordered"] else sorted(ids)
    if answer["kind"] == "categories":
        return answer["items"]
    return [answer["value"]]


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda c: f"{c['question_id']}-{c['tokens'] // 1000}k")
def test_answer_matches_independent_recomputation(case):
    text, _ = context_slice(case["tokens"])
    apps = parse(text)
    want = expected(case["question_id"], apps)
    if case["answer"]["kind"] == "number":
        assert abs(case["answer"]["value"] - float(want[0])) < 1e-9
    else:
        assert published(case["answer"]) == want
    # App names in the answer must be the names the model was shown.
    shown = {a["id"]: a["name"] for a in apps}
    for item in case["answer"].get("items", []):
        if isinstance(item, dict):
            assert shown[item["id"]] == item["name"]


def test_dataset_covers_every_question_and_size():
    seen = {(c["question_id"], c["tokens"]) for c in DATASET["cases"]}
    qids = {c["question_id"] for c in DATASET["cases"]}
    assert len(qids) == 20
    assert seen == {(q, t) for q in qids for t in SIZES}


def test_context_parses_back_to_every_row():
    for tokens in SIZES:
        text, rows = context_slice(tokens)
        assert len(parse(text)) == len(rows)
