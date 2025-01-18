def visualize_results():
    """
    Generate and save visualization plots for model evaluation results:

    1) Average score by model
    2) Score breakdown by difficulty level
    3) Score breakdown by token size
    4) Price vs. Performance scatter plot

    All plots are saved to the data folder for README reference and displayed in notebooks.
    """
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import numpy as np

    # Create data/plots directory if it doesn't exist
    Path("data/plots").mkdir(parents=True, exist_ok=True)

    # --- Load the JSON data ---
    with open("data/results.json", "r") as f:
        raw_data = json.load(f)

    # --- Flatten the JSON into a list of dictionaries ---
    data_for_df = []
    for model_name, model_data in raw_data.items():
        # Get pricing info
        pricing = model_data.get("pricing", {})
        prompt_price = pricing.get("prompt", 0)
        completion_price = pricing.get("completion", 0)
        total_price = prompt_price + completion_price

        for entry in model_data["results"]:
            data_for_df.append({
                "model_name": model_name,
                "question": entry.get("question", ""),
                "tokens": entry.get("tokens", 0),
                "difficulty": entry.get("difficulty", ""),
                "score": entry.get("score", 0.0),
                "price": total_price
            })

    # --- Convert to DataFrame ---
    df = pd.DataFrame(data_for_df)

    # --- 1) Average score by model with and without partial credit ---
    # Calculate strict (perfect) scores and partial-credit scores
    df['strict_score'] = (df['score'] == 1.0).astype(float)
    model_scores = df.groupby('model_name').agg({
        'score': 'mean',
        'strict_score': 'mean'
    }).sort_values('score', ascending=True)

    x = np.arange(len(model_scores.index))

    # Convert scores to percentages
    strict_vals = model_scores['strict_score'] * 100
    partial_addon_vals = (model_scores['score'] - model_scores['strict_score']) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the baseline (strict portion) in pink
    rects_strict = ax.barh(
        x,
        strict_vals,
        color='lightcoral',
        label='Without Partial Credit'
    )

    # Plot the additional partial-credit portion in sky-blue, stacked on top
    rects_partial = ax.barh(
        x,
        partial_addon_vals,
        left=strict_vals,
        color='skyblue',
        label='With Partial Credit'
    )

    ax.set_title("Average Score by Model", pad=20)
    ax.set_xlabel("Score (%)")
    ax.set_yticks(x)
    ax.set_yticklabels(model_scores.index)
    ax.set_xlim(0, 100)  # Set x-axis limits from 0 to 100%
    ax.legend()

    # Function to label the total score at the end of the stacked bar
    def autolabel_total(rects1, rects2):
        for r1, r2 in zip(rects1, rects2):
            w1 = r1.get_width()
            w2 = r2.get_width()
            
            total = w1 + w2
            y_pos = r1.get_y() + r1.get_height() / 2
            
            ax.text(
                total,
                y_pos,
                f"{total:.1f}%",  # Add % symbol
                va='center',
                ha='left',
                fontsize=8
            )

    # Call the labeling function
    autolabel_total(rects_strict, rects_partial)

    plt.tight_layout()
    plt.savefig('data/plots/1_average_scores.png',
                bbox_inches='tight', dpi=300)
    plt.show()


    # --- 2) Difficulty breakdown ---
    plt.figure(figsize=(12, 6))

    difficulty_order = ["Very Hard", "Hard", "Medium", "Easy"]

    # Convert scores to percentages
    diff_scores = df.pivot_table(
        index='difficulty',
        columns='model_name',
        values='score',
        aggfunc='mean'
    ).reindex(difficulty_order) * 100

    diff_scores.plot(kind='barh', width=0.8)
    plt.title("Model Performance by Difficulty", pad=20)
    plt.ylabel("Difficulty Level")
    plt.xlabel("Score (%)")
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100%
    plt.legend(title="Model", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig('data/plots/2_difficulty_breakdown.png', bbox_inches='tight', dpi=300)
    plt.show()


    # --- 3) Token size breakdown ---
    plt.figure(figsize=(12, 6))

    bins = [0, 10000, 20000, 30000, 40000, 50000,
            60000, 70000, 80000, 90000, 100000]
    labels = ['10k', '20k', '30k', '40k', '50k',
            '60k', '70k', '80k', '90k', '100k']

    df["token_bracket"] = pd.cut(
        df["tokens"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Convert scores to percentages
    token_scores = df.pivot_table(
        index='token_bracket',
        columns='model_name',
        values='score',
        aggfunc='mean'
    ) * 100

    # Reverse order so "10k" is at the top and "100k" at the bottom
    bracket_order = ['100k', '90k', '80k', '70k', '60k',
                    '50k', '40k', '30k', '20k', '10k']
    token_scores = token_scores.reindex(bracket_order)

    # Drop any bracket rows that are entirely empty
    token_scores.dropna(how='all', inplace=True)

    token_scores.plot(kind='barh', width=0.8)
    plt.title("Model Performance by Token Size", pad=20)
    plt.ylabel("Token Bracket")
    plt.xlabel("Score (%)")
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100%
    plt.legend(title="Model", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig('data/plots/3_token_size_breakdown.png', bbox_inches='tight', dpi=300)
    plt.show()


    # --- 4) Price vs Performance scatter plot ---
    plt.figure(figsize=(10, 6))

    # Calculate average score and price by model
    model_metrics = df.groupby('model_name').agg({
        'score': 'mean',
        'price': 'first'  # Price is same for all entries of a model
    }).reset_index()

    # Create scatter plot
    plt.scatter(model_metrics['price'], model_metrics['score'], alpha=0.6)

    # Add labels for each point
    for idx, row in model_metrics.iterrows():
        plt.annotate(
            # Use only the model name without provider
            row['model_name'].split('/')[-1],
            (row['price'], row['score']),
            xytext=(5, 5),
            textcoords='offset points'
        )

    plt.title("Price vs Performance Trade-off", pad=20)
    plt.xlabel("Price (USD per 1k tokens)")
    plt.ylabel("Average Score")

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/plots/4_price_performance.png',
                bbox_inches='tight', dpi=300)
    plt.show()  # Show in notebook

    # Print summary statistics
    print("\n=== Model Performance Summary ===")
    summary = df.groupby("model_name").agg({
        "score": ["mean", "std", "count"],
        "price": "first"
    }).round(3)
    print(summary)
