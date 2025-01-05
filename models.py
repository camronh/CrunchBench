# DF Model

from pydantic import BaseModel
from typing import TypedDict

class App(BaseModel):
    id: str
    name: str
    size: float
    currency: str
    price: float
    rating_count_tot: int
    user_rating: float
    ver: str
    prime_genre: str
    app_desc: str


class Example(TypedDict):
    question: str
    difficulty: str
    tokens: int


class ReferenceOutput(TypedDict):
    ground_truth: str

class Output(TypedDict):
    answer: str
