"""The published question/answer dataset (data/eval_dataset.json)."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from crunchbench.data import DATA_DIR, context_slice
from crunchbench.questions import QUESTIONS, RULES

DATASET_PATH = DATA_DIR / "eval_dataset.json"


class AmbiguousGroundTruth(ValueError):
    pass


def context_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_cases(context_sizes: list[int]) -> list[dict]:
    """Compute every (question, context size) case, refusing any ambiguous answer."""
    cases, problems = [], []
    for tokens in context_sizes:
        text, rows = context_slice(tokens)
        for q in QUESTIONS:
            answer = q.compute(rows)
            problems += [f"{q.id} @ {tokens}: {issue}" for issue in answer.issues]
            cases.append(
                {
                    "question_id": q.id,
                    "difficulty": q.difficulty,
                    "tokens": tokens,
                    "num_apps": len(rows),
                    "context_sha256": context_hash(text),
                    "question": q.text,
                    "reference": answer.render(),
                    "answer": answer.to_json(),
                }
            )
    if problems:
        raise AmbiguousGroundTruth("Ambiguous ground truth:\n" + "\n".join(problems))
    return cases


def write_dataset(context_sizes: list[int]) -> dict:
    dataset = {"rules": RULES, "cases": build_cases(context_sizes)}
    DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    return dataset


@lru_cache(maxsize=1)
def load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text())
