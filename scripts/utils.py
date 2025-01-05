from typing import List, Tuple
import pandas as pd
import tiktoken


def row_to_string(row: pd.Series) -> str:
    """Convert a row to a string representation.

    Args:
        row: A pandas Series representing a row from the app store dataset

    Returns:
        A formatted string containing the app's information
    """
    app_string = f"""App Name: {row['name']}
ID: {row.id}
Size: {round(row.size, 2)} MB
Price: {row.price} {row.currency}
Rating Count: {row.rating_count_tot}
Average User Rating: {row.user_rating}
Version: {row.ver}
Category: {row.prime_genre}
Description: {row.app_desc}"""
    return app_string


def df_to_string(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a string by converting each row.

    Args:
        df: A pandas DataFrame containing app store data

    Returns:
        A string containing all apps' information, separated by newlines
    """
    return "\n\n==============\n\n".join(df.apply(row_to_string, axis=1))


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for
        encoding_name: The name of the tiktoken encoding to use

    Returns:
        The number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def slice_dataframe_by_tokens(df: pd.DataFrame, token_limit: int) -> Tuple[str, pd.DataFrame]:
    """Slice a DataFrame based on token count and return both string and DataFrame.

    Args:
        df: The DataFrame to slice
        token_limit: Maximum number of tokens to include

    Returns:
        A tuple containing:
        - The string representation of the sliced DataFrame
        - The sliced DataFrame itself

    Example:
        >>> text, df_slice = slice_dataframe_by_tokens(df, 10000)
        >>> print(f"Slice contains {len(df_slice)} rows")
    """
    total_tokens = 0
    rows_text = []
    cutoff_idx = 0

    for idx, row in df.iterrows():
        row_text = row_to_string(row)
        row_tokens = count_tokens(row_text)

        if total_tokens + row_tokens > token_limit:
            break

        total_tokens += row_tokens
        rows_text.append(row_text)
        cutoff_idx = idx

    full_text = "\n\n==============\n\n".join(rows_text)
    df_slice = df.loc[:cutoff_idx].copy()

    return full_text, df_slice


def load_data(path: str = "data/appstore_preprocessed.csv") -> pd.DataFrame:
    """Load the appstore dataset from a CSV file."""
    return pd.read_csv(path)
