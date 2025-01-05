import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
from utils import slice_dataframe_by_tokens


def load_data() -> pd.DataFrame:
    """Load the App Store data from local files."""
    data_dir = Path("data")

    # Load main app data
    apps_df = pd.read_csv(data_dir / "AppleStore.csv")

    # Load descriptions
    desc_df = pd.read_csv(data_dir / "appleStore_description.csv")

    full_app_df = pd.merge(apps_df, desc_df, on='id', how='left')

    return full_app_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    # Select the columns we want
    new_df = df[['id', 'track_name_x', 'size_bytes_x', 'currency',
                 'price', 'rating_count_tot', 'user_rating', 'ver', 'prime_genre', 'app_desc']]

    # Rename the columns
    new_df = new_df.rename(
        columns={'track_name_x': 'name', 'size_bytes_x': 'size'})

    new_df['size'] = new_df['size'] / (1024 * 1024)  # Convert to MB
    new_df['name'] = new_df['name'].str.replace(
        r"[^a-zA-Z0-9\s]+", "", regex=True)
    new_df = new_df[new_df['name'].str.strip() != ""]
    new_df = new_df[new_df['rating_count_tot'] != 0]
    new_df = new_df[new_df['app_desc'].str.contains(
        r'[\u4e00-\u9fff]') == False]

    # Remove apps with descriptions longer than 1200 characters
    new_df = new_df[new_df['app_desc'].str.len() < 1200]

    # Sort by rating count (most reviewed first)
    new_df = new_df.sort_values("rating_count_tot", ascending=False)

    return new_df


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Truncating to token limit...")
    _, top_apps = slice_dataframe_by_tokens(df, token_limit=150000)

    # Sort by ID
    top_apps = top_apps.sort_values("id")

    print("Saving preprocessed data...")
    top_apps.to_csv("data/appstore_preprocessed.csv", index=False)

    print(f"Saved {len(top_apps)} rows to data/appstore_preprocessed.csv")


if __name__ == "__main__":
    main()
