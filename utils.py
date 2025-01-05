from typing import List, Tuple
import pandas as pd
import tiktoken
import requests
from typing import Dict, Optional, Tuple

dataset_name = "CrunchBench"

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


def store_results(results, model_name: str, custom_label: str = None, output_path: str = "data/results.json") -> None:
    """Store evaluation results in a JSON file, organizing them by model name.

    Args:
        results: Evaluation results to store
        model_name: Name/ID of the model (e.g., 'anthropic/claude-3-opus-20240229')
        custom_label: Optional custom label for the results
        output_path: Path to save results JSON file
    """
    import json
    import os
    from pathlib import Path

    df = pd.DataFrame(results)

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)

    # Try to get model pricing
    try:
        prompt_price, completion_price = get_model_pricing(model_name)
    except ValueError:
        # If pricing fetch fails, use None values
        prompt_price, completion_price = None, None

    # Convert DataFrame results to dictionary format
    new_results = []
    for _, row in df.iterrows():
        tokens = row["example"].inputs["tokens"]
        question = row["example"].inputs["question"]
        difficulty = row["example"].inputs["difficulty"]
        score = row["evaluation_results"]["results"][0].score

        new_results.append({
            "tokens": tokens,
            "question": question,
            "difficulty": difficulty,
            "score": score
        })

    # Store results with pricing info formatted as decimal strings
    result_entry = {
        "results": new_results,
        "pricing": {
            "prompt": "{:.8f}".format(prompt_price) if prompt_price is not None else None,
            "completion": "{:.8f}".format(completion_price) if completion_price is not None else None
        }
    }

    if custom_label:
        existing_results[custom_label] = result_entry
    else:
        existing_results[model_name] = result_entry

    # Write back to file
    with open(output_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

    print(f"Results for {model_name} saved!")




def get_model_pricing(model_name: str) -> Tuple[float, float]:
    """Fetch prompt and completion token prices for a specific model from OpenRouter API.
    
    Args:
        model_name: The model identifier (e.g., 'anthropic/claude-3-opus-20240229')
    
    Returns:
        Tuple of (prompt_price, completion_price) in USD per 1000 tokens
        
    Raises:
        ValueError: If model not found or API request fails
    """
    # OpenRouter API endpoint for model information
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        # Make GET request to OpenRouter API
        response = requests.get(
            url,
            params={"supported_parameters": "temperature"},
            timeout=10  # 10 second timeout
        )
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        # Find the matching model in the response
        models_data = response.json().get("data", [])
        for model in models_data:
            if model["id"] == model_name:
                pricing = model.get("pricing", {})
                # Format prices as decimal strings with proper precision
                prompt_price = "{:.8f}".format(float(pricing.get("prompt", "0")))
                completion_price = "{:.8f}".format(float(pricing.get("completion", "0")))
                return float(prompt_price), float(completion_price)
                
        # If we get here, model wasn't found
        raise ValueError(f"Model '{model_name}' not found in OpenRouter API response")
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch model pricing: {str(e)}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing API response: {str(e)}")