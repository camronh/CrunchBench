"""Build data/appstore_preprocessed.csv from the raw Kaggle files.

Raw data: https://www.kaggle.com/datasets/ramamet4/app-store-apple-data-set-10k-apps
Put AppleStore.csv and appleStore_description.csv in data/ and run:

    uv run scripts/preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crunchbench.config import load_settings
from crunchbench.data import DATA_DIR, ENCODING, PREPROCESSED_CSV, row_to_string

# Long descriptions keep their opening and lose the tail, which is mostly legal
# notices, subscription terms and press quotes.
MAX_DESC_CHARS = 1200
# Don't cut earlier than this when looking for a clean break.
MIN_DESC_CHARS = 800
ELLIPSIS = " [...]"


def shorten(text: str) -> str:
    """Cut a description to at most MAX_DESC_CHARS, at a line or sentence break if possible."""
    if len(text) <= MAX_DESC_CHARS:
        return text
    head = text[: MAX_DESC_CHARS - len(ELLIPSIS)]
    cut = max(head.rfind("\n"), *(head.rfind(p) + 1 for p in (". ", "! ", "? ")))
    if cut < MIN_DESC_CHARS:
        cut = head.rfind(" ")
    return head[:cut].rstrip() + ELLIPSIS


def main() -> None:
    apps = pd.read_csv(DATA_DIR / "AppleStore.csv")
    desc = pd.read_csv(DATA_DIR / "appleStore_description.csv")
    df = apps.merge(desc, on="id", how="left")

    df = df[
        [
            "id",
            "track_name_x",
            "size_bytes_x",
            "currency",
            "price",
            "rating_count_tot",
            "user_rating",
            "ver",
            "prime_genre",
            "app_desc",
        ]
    ]
    df = df.rename(columns={"track_name_x": "name", "size_bytes_x": "size"})
    df["size"] = df["size"] / (1024 * 1024)
    # Keep names plain ASCII so they are easy to quote back, and collapse the
    # gaps left behind by removed punctuation.
    df["name"] = (
        df["name"].str.replace(r"[^a-zA-Z0-9\s]+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df[df["name"] != ""]
    df = df[df["rating_count_tot"] != 0]
    df = df[~df["app_desc"].str.contains(r"[一-鿿]", na=False)]
    df["app_desc"] = df["app_desc"].map(shorten)

    # Most-reviewed first, then take as many as fit in the largest context size.
    # Every context size is then a prefix of the same pool.
    pool_tokens = max(load_settings().context_sizes)
    df = df.sort_values("rating_count_tot", ascending=False, kind="mergesort")
    enc = tiktoken.get_encoding(ENCODING)
    total, keep = 0, []
    for idx, row in df.iterrows():
        tokens = len(enc.encode(row_to_string(row)))
        if total + tokens > pool_tokens:
            break
        total += tokens
        keep.append(idx)

    # The model sees apps in ID order, which mixes popular and niche apps.
    out = df.loc[keep].sort_values("id")
    out.to_csv(PREPROCESSED_CSV, index=False)
    print(f"Saved {len(out)} apps ({total:,} tokens) to {PREPROCESSED_CSV.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
