"""CrunchBench questions and their ground-truth computations.

Every question is computed over exactly the apps shown to the model at a given
context size. Each computation also reports "issues": anything that would make
the ground truth ambiguous, such as ties at a ranking cutoff, a value sitting on
a rounding boundary, or a keyword whose matches depend on how "word" is read.
`scripts/build_dataset.py` refuses to write a dataset while any issue remains.

Shared definitions the model is given live in `RULES` and are sent with every
question, so the wording of each question can stay natural.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import pandas as pd

RULES = """\
- "Rating" means the Average User Rating field. "Rating count" means the Rating Count field.
- A free app has a price of 0.0. A paid app has a price greater than 0.
- "Size" means the Size field in MB.
- A description "contains the word X" when X appears in the Description as a whole word, ignoring case.
- Category names are the exact values of the Category field.
- Identify each app by its ID and name.
- Use every app in the data above, and only those apps.
- If fewer items qualify than the question asks for, list only the ones that qualify. If none qualify, return an empty list."""

# What kind of value a question expects. It picks the response schema and the grader.
APPS, CATEGORIES, COUNT, NUMBER = "apps", "categories", "count", "number"


@dataclass
class Answer:
    """A ground-truth answer plus any reasons it might be ambiguous.

    `items` holds {"id", "name"} dicts for apps, strings for categories, one int
    for a count, or one float for a number.
    """

    kind: str
    items: list
    ordered: bool = True
    decimals: int | None = None
    prefix: str = ""
    issues: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable form, shown as the EZVals reference and in questions.md."""
        if self.kind == NUMBER:
            return f"{self.prefix}{self.items[0]:.{self.decimals}f}"
        if self.kind == COUNT:
            return str(self.items[0])
        if not self.items:
            return "None (empty list)"
        labels = [f"{i['name']} (ID {i['id']})" if self.kind == APPS else i for i in self.items]
        if len(labels) == 1 or not self.ordered:
            return "\n".join(labels)
        return "\n".join(f"{n}. {label}" for n, label in enumerate(labels, 1))

    def to_json(self) -> dict:
        """The machine-readable answer the grader compares against."""
        data: dict = {"kind": self.kind, "ordered": self.ordered and len(self.items) > 1}
        if self.kind in (APPS, CATEGORIES):
            data["items"] = self.items
        else:
            data["value"] = self.items[0]
        if self.decimals is not None:
            data["decimals"] = self.decimals
        return data


@dataclass(frozen=True)
class Question:
    id: str
    difficulty: str
    text: str
    compute: Callable[[pd.DataFrame], Answer]


# ---------------------------------------------------------------------------
# Helpers. Each one checks for the ambiguity it could introduce.
# ---------------------------------------------------------------------------

FREE = lambda df: df[df["price"] == 0]
PAID = lambda df: df[df["price"] > 0]


def contains_word(df: pd.DataFrame, word: str) -> tuple[pd.Series, list[str]]:
    """Whole-word, case-insensitive match on the description.

    Flags the keyword if a looser reading (plain substring, or allowing a plural)
    would select a different set of apps.
    """
    desc = df["app_desc"]
    whole = desc.str.contains(rf"\b{re.escape(word)}\b", case=False, regex=True)
    substring = desc.str.lower().str.contains(word.lower(), regex=False)
    plural = desc.str.contains(rf"\b{re.escape(word)}(?:s|es)?\b", case=False, regex=True)
    issues = []
    if not (whole.equals(substring) and whole.equals(plural)):
        issues.append(f"keyword '{word}' matches differ between whole-word, substring and plural readings")
    return whole, issues


def app_items(df: pd.DataFrame) -> list[dict]:
    return [{"id": int(r.id), "name": r.name} for r in df[["id", "name"]].itertuples(index=False)]


def rank(df: pd.DataFrame, keys: list[tuple[str, bool]], k: int, kind: str = APPS) -> Answer:
    """Top-k rows ordered by `keys` (column, ascending).

    Flags a tie if the full sort key cannot separate the top k from each other
    or from the (k+1)th row.
    """
    cols = [c for c, _ in keys]
    ordered = df.sort_values(cols, ascending=[a for _, a in keys], kind="mergesort")
    head = ordered.head(k + 1)
    issues = []
    key_tuples = [tuple(r) for r in head[cols].itertuples(index=False)]
    if len(set(key_tuples)) < len(key_tuples):
        issues.append(f"tie in sort key {cols} within the top {k}+1")
    top = ordered.head(k)
    items = app_items(top) if kind == APPS else top["prime_genre"].tolist()
    return Answer(kind=kind, items=items, issues=issues)


def pick_group(values: pd.Series, largest: bool = True, k: int = 1, min_gap: float = 1e-9) -> Answer:
    """Pick the top-k categories from a Series indexed by category.

    Flags a tie if neighbouring values within the top k+1 are closer than `min_gap`.
    """
    s = values.sort_values(ascending=not largest)
    issues = []
    top = s.head(k + 1).tolist()
    for a, b in pairwise(top):
        if abs(a - b) < min_gap:
            issues.append(f"group values too close to rank: {s.head(k + 1).round(6).to_dict()}")
            break
    return Answer(kind=CATEGORIES, items=[str(x) for x in s.head(k).index], issues=issues)


def number(value: float, decimals: int, prefix: str = "") -> Answer:
    """A number the question asks for at `decimals` places.

    The exact value is stored. The grader accepts anything within half a unit of
    the last decimal place, so both rounding directions count when the exact
    value sits on a .5 boundary.
    """
    return Answer(kind=NUMBER, items=[float(value)], decimals=decimals, prefix=prefix)


def count(n: int) -> Answer:
    return Answer(kind=COUNT, items=[int(n)])


# ---------------------------------------------------------------------------
# Easy: one filter or one sort.
# ---------------------------------------------------------------------------


def e1(df):
    return rank(df, [("rating_count_tot", False)], 3)


def e2(df):
    return rank(df, [("size", False)], 1)


def e3(df):
    return rank(df[df["prime_genre"] == "Games"], [("rating_count_tot", False)], 5)


def e4(df):
    return count((df["prime_genre"] == "Games").sum())


def e5(df):
    top = df[df["price"] == df["price"].max()].sort_values("id")
    return Answer(kind=APPS, items=app_items(top), ordered=False)


# ---------------------------------------------------------------------------
# Medium: two conditions, a tie-breaker, or a simple aggregate.
# ---------------------------------------------------------------------------


def m1(df):
    return rank(FREE(df), [("user_rating", False), ("rating_count_tot", False)], 3)


def m2(df):
    return rank(PAID(df), [("user_rating", True), ("rating_count_tot", True)], 5)


def m3(df):
    return count(((df["price"] == 0) & (df["user_rating"] >= 4.5)).sum())


def m4(df):
    return pick_group(PAID(df).groupby("prime_genre")["rating_count_tot"].sum())


def m5(df):
    mask, issues = contains_word(df, "offline")
    ans = rank(df[mask], [("rating_count_tot", False)], 3)
    ans.issues += issues
    return ans


# ---------------------------------------------------------------------------
# Hard: derived metrics, grouped aggregates, several combined conditions.
# ---------------------------------------------------------------------------


def h1(df):
    paid = PAID(df).assign(ratio=lambda d: d["rating_count_tot"] / d["price"])
    return rank(paid, [("ratio", False)], 3)


def h2(df):
    groups = df.groupby("prime_genre")["user_rating"].agg(["mean", "size"])
    return pick_group(groups[groups["size"] >= 6]["mean"], k=3)


def h3(df):
    games = PAID(df)[lambda d: d["prime_genre"] == "Games"]
    return number(games["price"].mean(), 2, prefix="$")


def h4(df):
    games = df[df["prime_genre"] == "Games"]
    multi, i1 = contains_word(games, "multiplayer")
    ach, i2 = contains_word(games, "achievements")
    ans = rank(games[multi | ach], [("user_rating", False), ("rating_count_tot", False)], 3)
    ans.issues += i1 + i2
    return ans


def h5(df):
    big = df[(df["size"] > 200) & (df["user_rating"] >= 4.5)]
    ans = rank(big, [("size", False)], 10)
    # Sizes are shown rounded to 2 decimals; make sure that rounding cannot change
    # who clears the 200 MB bar or the order of the top 10 (and the 11th).
    if ((df["size"].round(2) > 200) != (df["size"] > 200)).any():
        ans.issues.append("an app's displayed size rounds across the 200 MB threshold")
    if big.nlargest(11, "size")["size"].round(2).duplicated().any():
        ans.issues.append("two of the largest qualifying apps show the same rounded size")
    return ans


# ---------------------------------------------------------------------------
# Very hard: multi-step reasoning over groups, with derived thresholds.
# ---------------------------------------------------------------------------


def v1(df):
    paid = PAID(df)
    g = paid.groupby("prime_genre").agg(n=("price", "size"), r=("user_rating", "mean"), p=("price", "mean"))
    g = g[g["n"] >= 3]
    return pick_group(g["r"] / g["p"])


def v2(df):
    g = df.groupby("prime_genre")["user_rating"].agg(n="size", sd=lambda s: s.std(ddof=0))
    return pick_group(g[g["n"] >= 5]["sd"], min_gap=1e-4)


def v3(df):
    top = df[df["user_rating"] >= 4.5]
    g = top.groupby("prime_genre").agg(n=("id", "size"), reviews=("rating_count_tot", "sum")).reset_index()
    return rank(g, [("n", False), ("reviews", False)], 3, kind=CATEGORIES)


def v4(df):
    big = df[df["size"] > 100]
    g = big.groupby("prime_genre")["rating_count_tot"].agg(["mean", "size"])
    ans = pick_group(g[g["size"] >= 3]["mean"])
    if ((df["size"].round(2) > 100) != (df["size"] > 100)).any():
        ans.issues.append("an app's displayed size rounds across the 100 MB threshold")
    return ans


def v5(df):
    # Ratings are multiples of 0.5, so compare in exact integer half-stars:
    # rating > sum/n  <=>  rating * n > sum. No floating point involved.
    halves = (df["user_rating"] * 2).round().astype(int)
    assert ((halves / 2) == df["user_rating"]).all(), "ratings are not multiples of 0.5"
    grp = df["prime_genre"]
    cat_sum = halves.groupby(grp).transform("sum")
    cat_n = halves.groupby(grp).transform("size")
    above = df[halves * cat_n > cat_sum]
    return rank(above, [("rating_count_tot", False)], 3)


QUESTIONS: list[Question] = [
    Question("E1", "Easy", "Which 3 apps have the highest rating count? List them from highest to lowest.", e1),
    Question("E2", "Easy", "Which app has the largest size?", e2),
    Question(
        "E3",
        "Easy",
        "In the 'Games' category, which 5 apps have the highest rating count? List them from highest to lowest.",
        e3,
    ),
    Question("E4", "Easy", "How many apps are in the 'Games' category?", e4),
    Question("E5", "Easy", "Which app or apps have the highest price? List every app at that price.", e5),
    Question(
        "M1",
        "Medium",
        "Among free apps, which 3 have the highest rating? Break ties by higher rating count. List them in order.",
        m1,
    ),
    Question(
        "M2",
        "Medium",
        "Among paid apps, which 5 have the lowest rating? Break ties by lower rating count. List them in order, lowest first.",
        m2,
    ),
    Question("M3", "Medium", "How many free apps have a rating of 4.5 or higher?", m3),
    Question("M4", "Medium", "Which category has the highest total rating count across its paid apps?", m4),
    Question(
        "M5",
        "Medium",
        "Among apps whose description contains the word 'offline', which 3 have the highest rating count? List them from highest to lowest.",
        m5,
    ),
    Question(
        "H1",
        "Hard",
        "Among paid apps, which 3 have the highest ratio of rating count to price? List them from highest ratio to lowest.",
        h1,
    ),
    Question(
        "H2",
        "Hard",
        "Considering only categories with at least 6 apps, which 3 categories have the highest mean rating? List them from highest to lowest.",
        h2,
    ),
    Question(
        "H3", "Hard", "What is the mean price of the paid apps in the 'Games' category? Round to 2 decimal places.", h3
    ),
    Question(
        "H4",
        "Hard",
        "In the 'Games' category, among apps whose description contains the word 'multiplayer' or the word 'achievements', which 3 have the highest rating? Break ties by higher rating count. List them in order.",
        h4,
    ),
    Question(
        "H5",
        "Hard",
        "Among apps larger than 200 MB with a rating of 4.5 or higher, which 10 are the largest? List them from largest to smallest.",
        h5,
    ),
    Question(
        "V1",
        "Very Hard",
        "For each category with at least 3 paid apps, divide the mean rating of its paid apps by the mean price of its paid apps. Which category has the highest result?",
        v1,
    ),
    Question(
        "V2",
        "Very Hard",
        "Considering only categories with at least 5 apps, which category has the largest population standard deviation of rating?",
        v2,
    ),
    Question(
        "V3",
        "Very Hard",
        "Which 3 categories have the most apps with a rating of 4.5 or higher? Break ties by the higher combined rating count of those apps. List them in order.",
        v3,
    ),
    Question(
        "V4",
        "Very Hard",
        "Among apps larger than 100 MB, which category has the highest mean rating count? Only consider categories with at least 3 such apps.",
        v4,
    ),
    Question(
        "V5",
        "Very Hard",
        "Among apps whose rating is strictly higher than the mean rating of all apps in their own category, which 3 have the highest rating count? List them from highest to lowest.",
        v5,
    ),
]

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}
