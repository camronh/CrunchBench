from llms import openrouter
from typing import Literal
from pydantic import BaseModel, Field
from langsmith.schemas import Example


class CorrectnessScore(BaseModel):
    reasoning: str = Field(description="The reasoning behind the score")
    score: Literal["incorrect", "partially correct", "correct"] = Field(
        description="The score of the answer")


def correctness_judge(example: Example, outputs: dict) -> dict:
    """Judge the correctness of the output."""

    question = example.inputs["question"]
    correct_answer = example.outputs["ground_truth"]
    llm_answer = outputs["answer"] or "None Found"

    prompt = f"""You are an impartial judge, comparing the LLM's answer to a ground truth. Your job is to determine the correctness of the LLM's answer by providing two fields in your response:
- reasoning: Your thought process
- score: One of the following strings: “correct”, “partially correct”, or “incorrect”

Here are the scoring rules:

1) "correct" — if the LLM's answer includes all apps or categories in the correct order, without introducing any new ones that aren't in the ground truth. Having extra descriptive detail on the same items is acceptable.
2) "partially correct" — if the LLM's answer includes at least one correct app or category from the ground truth (order doesn't matter), but it isn't fully correct (for instance, missing some items or adding extra ones).
3) "incorrect" — if none of the items in the LLM's answer match the ground truth.

Below are various scenarios:

Scenario 1:
- The correct answer has exactly two items. 
- The LLM's answer has those two items in the same order with no extras.
- Score = “correct”.

Scenario 2:
- The correct answer has two items.
- The LLM's answer contains those two items plus an additional item not in the ground truth.
- Score = “partially correct” (because of the extra item).

Scenario 3:
- The correct answer has multiple items.
- The LLM's answer includes at least one of the correct items but is missing the rest.
- Score = “partially correct”.

Scenario 4:
- The correct answer has multiple items.
- The LLM's answer shares no items with the ground truth.
- Score = “incorrect”.

Scenario 5:
- The correct answer has multiple items in a specific sequence.
- The LLM's answer has those items but in a different order (no extras).
- Score = “partially correct” (the order is important, so it's not fully correct).

Scenario 6:
- The correct answer is empty or 'None Found'.
- The LLM's answer that none were found.
- Score = “correct” (both have nothing).

Scenario 7:
- The correct answer is empty or 'None Found'.
- The LLM's answer includes one or more items.
- Score = “incorrect” (it introduced items when there were none in the ground truth).

<Question>
{question}
</Question>

<Correct Answer>
{correct_answer}
</Correct Answer>

<LLM Answer>
{llm_answer}
</LLM Answer>
"""
    judge = openrouter.with_structured_output(CorrectnessScore)
    response: CorrectnessScore = judge.invoke(prompt)

    # Correct score = 1, Partial score = 0.5, Incorrect score = 0
    score = 1 if response.score == "correct" else 0.5 if response.score == "partially correct" else 0

    return {"key": "correct", "score": score, "comment": response.reasoning}
