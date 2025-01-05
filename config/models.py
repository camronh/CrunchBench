# DF Model

from pydantic import BaseModel


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
