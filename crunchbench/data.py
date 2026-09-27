"""Loading the preprocessed App Store data and turning it into model context."""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREPROCESSED_CSV = DATA_DIR / "appstore_preprocessed.csv"
APP_SEPARATOR = "\n\n==============\n\n"
ENCODING = "cl100k_base"


def row_to_string(row: pd.Series) -> str:
    """Render one app exactly the way the model sees it."""
    return (
        f"App Name: {row['name']}\n"
        f"ID: {row['id']}\n"
        f"Size: {round(row['size'], 2)} MB\n"
        f"Price: {row['price']} {row['currency']}\n"
        f"Rating Count: {row['rating_count_tot']}\n"
        f"Average User Rating: {row['user_rating']}\n"
        f"Version: {row['ver']}\n"
        f"Category: {row['prime_genre']}\n"
        f"Description: {row['app_desc']}"
    )


@lru_cache(maxsize=1)
def load_apps() -> pd.DataFrame:
    """Load the preprocessed dataset (sorted by app ID, the order the model sees)."""
    df = pd.read_csv(PREPROCESSED_CSV, dtype={"ver": str})
    df["name"] = df["name"].str.strip()
    return df.reset_index(drop=True)


@cache
def _row_tokens() -> tuple[int, ...]:
    enc = tiktoken.get_encoding(ENCODING)
    return tuple(len(enc.encode(row_to_string(r))) for _, r in load_apps().iterrows())


@cache
def context_slice(token_budget: int) -> tuple[str, pd.DataFrame]:
    """Return (context text, matching rows) for the first apps that fit in the budget.

    Apps are added in ID order until the next one would push the per-app token
    total past the budget. The rows returned are exactly the apps in the text,
    so ground truth is always computed over what the model actually sees.
    """
    total, n = 0, 0
    for tokens in _row_tokens():
        if total + tokens > token_budget:
            break
        total += tokens
        n += 1
    rows = load_apps().iloc[:n].copy()
    text = APP_SEPARATOR.join(row_to_string(r) for _, r in rows.iterrows())
    return text, rows


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding(ENCODING).encode(text))
