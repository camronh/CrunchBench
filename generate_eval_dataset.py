import pandas as pd
from typing import List, Optional
from utils import slice_dataframe_by_tokens, load_data
from pydantic import BaseModel
import uuid
import inspect
from types import FrameType
from langsmith import Client
from dotenv import load_dotenv
from utils import dataset_name

load_dotenv()

# Create a namespace UUID for our analytics benchmark
ANALYTICS_BENCHMARK_NAMESPACE = uuid.UUID(
    '6ba7b810-9dad-11d1-80b4-00c04fd430c8')


class Example(BaseModel):
    id: uuid.UUID
    question: str
    ground_truth: str
    difficulty: str
    tokens: int

    @classmethod
    def create(cls, question: str, ground_truth: str, difficulty: str, tokens: int) -> 'Example':
        """Create an Example with a deterministic UUID based on the calling method's name and token count."""
        # Get the calling function's name using inspect
        frame: Optional[FrameType] = inspect.currentframe()
        try:
            # Go up one frame to get the caller's name
            if frame is None:
                raise RuntimeError("Could not get current frame")
            caller_frame = frame.f_back
            if caller_frame is None:
                raise RuntimeError("Could not get caller frame")
            method_name = caller_frame.f_code.co_name
            # Generate a UUID5 from the method name combined with token count
            uuid_input = f"{method_name}_{tokens}"
            example_id = uuid.uuid5(ANALYTICS_BENCHMARK_NAMESPACE, uuid_input)
            return cls(
                id=example_id,
                question=question,
                ground_truth=ground_truth,
                difficulty=difficulty,
                tokens=tokens
            )
        finally:
            # Clean up frame references to prevent memory leaks
            del frame
            if 'caller_frame' in locals():
                del caller_frame


# Prompt template structured to take advantage of caching
system_prompt = """<APP STORE DATA>
{context_str}
</APP STORE DATA>

Your job is to analyze the <APP STORE DATA> and answer the following question as succinctly as possible based on the <APP STORE DATA> provided. Feel free to think and \
reason through the problem first but then provide your answer wrapped in <ANSWER> tags. Here is the question:

'{question}'
"""


def publish_dataset(examples: List[Example]):
    client = Client()

    # Check for dataset, create if not found
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name=dataset_name,
                              description="LLM App Analytics Questions")

    # Create examples
    for example in examples:
        shortened_tokens = f"{example.tokens // 1000}k"
        token_str = f"{shortened_tokens} Tokens"  # Ex: 10k Tokens
        client.create_example(
            dataset_name=dataset_name,
            example_id=example.id,
            inputs={
                "question": example.question,
                "difficulty": example.difficulty,
                "tokens": example.tokens,
            },
            outputs={"ground_truth": example.ground_truth},
            metadata={"difficulty": example.difficulty,
                      "tokens": example.tokens},
            split=[example.difficulty, token_str]
        )


def get_app_names(df: pd.DataFrame) -> str:
    """Return a numbered list of app names."""
    return "\n".join([f"{i+1}. {name}" for i, name in enumerate(df["name"].tolist())])


def get_categories_as_string(categories: pd.Series) -> str:
    """Return categories joined by newline."""
    return "\n".join(categories.tolist())


class GroundTruthGenerator:
    examples: List[Example] = []

    def __init__(self, full_df: pd.DataFrame = None, max_tokens: int = 120000):
        # If no DataFrame provided, load from CSV
        self.full_df = full_df if full_df is not None else load_data()
        self.max_tokens = max_tokens

    def generate_examples(self) -> List[Example]:
        step_size = 10000
        num_steps = self.max_tokens // step_size
        examples = []

        for i in range(num_steps):
            tokens_in_step = step_size * (i + 1)
            context_str, context_df = slice_dataframe_by_tokens(
                self.full_df, tokens_in_step
            )

            #
            # EASY (1–5)
            #
            examples.append(self._easy_top_3_avg_rating(
                context_df, context_str, tokens_in_step))

            examples.append(self._easy_app_most_reviews(
                context_df, context_str, tokens_in_step))

            examples.append(self._easy_top_5_prod_by_rating(
                context_df, context_str, tokens_in_step))

            examples.append(self._easy_top_3_free_apps(
                context_df, context_str, tokens_in_step))

            examples.append(self._easy_top_5_games_by_reviews(
                context_df, context_str, tokens_in_step))

            #
            # MEDIUM (6–10)
            #
            examples.append(self._medium_3_paid_productivity_under_299(
                context_df, context_str, tokens_in_step))

            examples.append(self._medium_category_highest_mean_rating(
                context_df, context_str, tokens_in_step))

            examples.append(self._medium_5_lowest_investment_15000_reviews_rating4(
                context_df, context_str, tokens_in_step))

            examples.append(self._medium_3_lifestyle_family_15000_reviews(
                context_df, context_str, tokens_in_step))

            examples.append(self._medium_5_lowest_paid_15000_reviews(
                context_df, context_str, tokens_in_step))

            #
            # HARD (11–15)
            #
            examples.append(self._hard_3_categories_100k_reviews_highest_mean(
                context_df, context_str, tokens_in_step))

            examples.append(self._hard_5_paid_fitness_diet_100k_reviews_4_5(
                context_df, context_str, tokens_in_step))

            examples.append(self._hard_3_paid_largest_ratio_reviews_price(
                context_df, context_str, tokens_in_step))

            examples.append(self._hard_single_cat_highest_total_reviews_free_apps(
                context_df, context_str, tokens_in_step))

            examples.append(self._hard_5_finance_stock_investment_rating_4_2(
                context_df, context_str, tokens_in_step))

            #
            # VERY HARD (16–20)
            #
            examples.append(self._veryhard_3_categories_cloud_15000_reviews_4_5_price_above1(
                context_df, context_str, tokens_in_step))

            examples.append(self._veryhard_5_top10pct_reviews_bottom10pct_price_prod_4_5(
                context_df, context_str, tokens_in_step))

            examples.append(self._veryhard_single_cat_kids_highest_mean_rating_price_ratio(
                context_df, context_str, tokens_in_step))

            examples.append(self._veryhard_stddev_top5_cat_above_0_99(
                context_df, context_str, tokens_in_step))

            examples.append(self._veryhard_3_apps_mention_two_of_learning_assistant_ai_study(
                context_df, context_str, tokens_in_step))

        self.examples = examples
        publish_dataset(examples)
        return examples

    # -------------------------------------------------------------------------
    # EASY (1–5)
    # -------------------------------------------------------------------------

    def _easy_top_3_avg_rating(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "What are the top 3 apps by average rating?"
        top_df = df.sort_values(by="user_rating", ascending=False).head(3)
        ground_truth = get_app_names(top_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Easy",
            tokens=tokens_in_step
        )

    def _easy_app_most_reviews(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "What app has the most reviews?"
        top_df = df.sort_values(by="rating_count_tot", ascending=False).head(1)
        ground_truth = top_df["name"].iloc[0]
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Easy",
            tokens=tokens_in_step
        )

    def _easy_top_5_prod_by_rating(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "In the 'Social Networking' category, what are the top 5 apps by average rating?"
        # Convert genre column to lowercase for case-insensitive matching
        df_copy = df.copy()
        df_copy["prime_genre"] = df_copy["prime_genre"].str.lower()
        prod_df = df_copy[df_copy["prime_genre"] == "social networking"].copy()
        sorted_df = prod_df.sort_values(
            by="user_rating", ascending=False).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Easy",
            tokens=tokens_in_step
        )

    def _easy_top_3_free_apps(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "Among free apps, what are the top 3 by average rating?"
        free_df = df[df["price"] == 0].copy()
        sorted_df = free_df.sort_values(
            by="user_rating", ascending=False).head(3)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Easy",
            tokens=tokens_in_step
        )

    def _easy_top_5_games_by_reviews(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "In the 'Games' category, what are the top 5 apps by number of reviews?"
        games_df = df[df["prime_genre"].str.lower() == "games"].copy()
        sorted_df = games_df.sort_values(
            by="rating_count_tot", ascending=False).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Easy",
            tokens=tokens_in_step
        )

    # -------------------------------------------------------------------------
    # MEDIUM (6–10)
    # -------------------------------------------------------------------------

    def _medium_3_paid_productivity_under_299(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "Among paid 'Entertainment' apps priced under $2.99, what are the top 3 by average rating?"
        filtered = df[
            (df["prime_genre"].str.lower() == "entertainment") &
            (df["price"] > 0) &
            (df["price"] < 2.99)
        ].copy()
        sorted_df = filtered.sort_values(
            by="user_rating", ascending=False).head(3)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Medium",
            tokens=tokens_in_step
        )

    def _medium_category_highest_mean_rating(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "What category has the highest mean rating across all its apps?"
        grouped = df.groupby("prime_genre")["user_rating"].mean().reset_index()
        highest_row = grouped.sort_values(
            by="user_rating", ascending=False).head(1)
        ground_truth = highest_row["prime_genre"].iloc[0]
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Medium",
            tokens=tokens_in_step
        )

    def _medium_5_lowest_investment_15000_reviews_rating4(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = ("What are the bottom 5 lowest-rated apps that explicitly mention the word 'investment' in their description, "
                    "have 15,000+ reviews, and have an average rating ≥ 4.0?")
        filtered = df[
            df["app_desc"].str.lower().str.contains("investment", na=False) &
            (df["rating_count_tot"] >= 15000) &
            (df["user_rating"] >= 4.0)
        ].copy()
        # "Bottom 5 lowest-rated" => sort ascending
        sorted_df = filtered.sort_values(
            by="user_rating", ascending=True).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Medium",
            tokens=tokens_in_step
        )

    def _medium_3_lifestyle_family_15000_reviews(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = ("In the 'Entertainment' category, what are the top 3 highest-rated apps that explicitly mention the word 'family' "
                    "in their description and have 100+ reviews?")
        ls_df = df[(df["prime_genre"].str.lower() == "entertainment")].copy()
        family_df = ls_df[ls_df["app_desc"].str.lower(
        ).str.contains("family", na=False)]
        sorted_df = family_df.sort_values(
            by="user_rating", ascending=False).head(3)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Medium",
            tokens=tokens_in_step
        )

    def _medium_5_lowest_paid_15000_reviews(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "Among paid apps with 15,000+ reviews, what are the 5 lowest-rated apps?"
        filtered = df[(df["price"] > 0) & (
            df["rating_count_tot"] >= 15000)].copy()
        sorted_df = filtered.sort_values(
            by="user_rating", ascending=True).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Medium",
            tokens=tokens_in_step
        )

    # -------------------------------------------------------------------------
    # HARD (11–15)
    # -------------------------------------------------------------------------

    def _hard_3_categories_100k_reviews_highest_mean(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "For apps with 100+ reviews, what are the top 3 categories by mean rating?"
        filtered = df[df["rating_count_tot"] >= 100].copy()
        grouped = filtered.groupby("prime_genre")[
            "user_rating"].mean().reset_index()
        sorted_df = grouped.sort_values(
            by="user_rating", ascending=False).head(3)
        ground_truth = "\n".join(
            [f"{i+1}. {category}" for i, category in enumerate(sorted_df["prime_genre"].tolist())])
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Hard",
            tokens=tokens_in_step
        )

    def _hard_5_paid_fitness_diet_100k_reviews_4_5(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = ("What are the top 5 highest-rated paid apps that have a rating < 4.5, have 10+ reviews, "
                    "and explicitly mention the words 'fitness' or 'diet' in their description?")
        filtered = df[
            (df["price"] > 0) &
            (df["user_rating"] < 4.5) &
            (df["rating_count_tot"] >= 10)
        ].copy()
        mask_fit = filtered["app_desc"].str.lower(
        ).str.contains("fitness", na=False)
        mask_diet = filtered["app_desc"].str.lower(
        ).str.contains("diet", na=False)
        final_df = filtered[mask_fit | mask_diet]
        sorted_df = final_df.sort_values(
            by="user_rating", ascending=False).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Hard",
            tokens=tokens_in_step
        )

    def _hard_3_paid_largest_ratio_reviews_price(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "Among paid apps, what are the top 3 by reviews-to-price ratio?"
        paid_df = df[df["price"] > 0].copy()
        paid_df["reviews_to_price"] = paid_df["rating_count_tot"] / \
            paid_df["price"]
        sorted_df = paid_df.sort_values(
            by="reviews_to_price", ascending=False).head(3)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Hard",
            tokens=tokens_in_step
        )

    def _hard_single_cat_highest_total_reviews_free_apps(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = "Which category has the highest total review count among free apps?"
        free_df = df[df["price"] == 0].copy()
        grouped = free_df.groupby("prime_genre")[
            "rating_count_tot"].sum().reset_index()
        sorted_df = grouped.sort_values(
            by="rating_count_tot", ascending=False).head(1)
        ground_truth = sorted_df["prime_genre"].iloc[0] if len(
            sorted_df) else "None"
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Hard",
            tokens=tokens_in_step
        )

    def _hard_5_finance_stock_investment_rating_4_2(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = ("In the 'Games' category, what are the top 5 highest-rated apps that explicitly mention the words 'controller' or 'story' "
                    "in name or description, are paid, and have an average rating < 4.2?")
        # Convert genre column to lowercase for case-insensitive matching
        df_copy = df.copy()
        df_copy["prime_genre"] = df_copy["prime_genre"].str.lower()
        finance_df = df_copy[(df_copy["prime_genre"] == "games") & (
            df_copy["price"] > 0) & (df_copy["user_rating"] < 4.2)]
        mask_stock_name = finance_df["name"].str.lower(
        ).str.contains("controller", na=False)
        mask_stock_desc = finance_df["app_desc"].str.lower(
        ).str.contains("controller", na=False)
        mask_invest_name = finance_df["name"].str.lower(
        ).str.contains("story", na=False)
        mask_invest_desc = finance_df["app_desc"].str.lower(
        ).str.contains("story", na=False)
        mention_mask = mask_stock_name | mask_stock_desc | mask_invest_name | mask_invest_desc
        filtered = finance_df[mention_mask]
        sorted_df = filtered.sort_values(
            by="user_rating", ascending=False).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Hard",
            tokens=tokens_in_step
        )

    # -------------------------------------------------------------------------
    # VERY HARD (16–20)
    # -------------------------------------------------------------------------

    def _veryhard_3_categories_cloud_15000_reviews_4_5_price_above1(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = (
            "For apps explicitly mentioning the word 'cloud' in their description, what are the top 3 categories by mean rating, "
            "considering only apps that have 15,000+ reviews, have a rating ≥ 4.5, and cost more than $1?"
        )
        filtered = df[
            df["app_desc"].str.lower().str.contains("cloud", na=False)
            & (df["rating_count_tot"] >= 15000)
            & (df["user_rating"] >= 4.5)
            & (df["price"] > 1)
        ].copy()
        grouped = filtered.groupby("prime_genre")[
            "user_rating"].mean().reset_index()
        sorted_df = grouped.sort_values(
            by="user_rating", ascending=False).head(3)
        ground_truth = "\n".join(
            [f"{i+1}. {category}" for i, category in enumerate(sorted_df["prime_genre"].tolist())])
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Very Hard",
            tokens=tokens_in_step
        )

    def _veryhard_5_top10pct_reviews_bottom10pct_price_prod_4_5(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = (
            "What are the top 5 highest-rated apps that are in the top 10% by review count, "
            "in the bottom 10% by price (excluding free apps), explicitly mention the word 'productivity' in their description, "
            "and have a rating ≥ 4.5?"
        )
        # top 10% by review
        review_thresh = df["rating_count_tot"].quantile(0.90)
        # bottom 10% by price, excluding free => must filter price>0 first
        paid_df = df[df["price"] > 0].copy()
        price_thresh = paid_df["price"].quantile(0.10)

        # Filter for productivity apps
        filtered = df[
            # 10% of apps by review count
            (df["rating_count_tot"] >= review_thresh)
            & (df["price"] > 0)  # paid apps
            & (df["price"] <= price_thresh)  # bottom 10% of paid apps by price
            # productivity apps
            & (df["app_desc"].str.lower().str.contains("productivity", na=False))
            & (df["user_rating"] >= 4.5)  # average rating >= 4.5
        ].copy()

        sorted_df = filtered.sort_values(
            by="user_rating", ascending=False).head(5)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Very Hard",
            tokens=tokens_in_step
        )

    def _veryhard_single_cat_kids_highest_mean_rating_price_ratio(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = (
            'Among apps explicitly mentioning the word "kids" in name or description, which category has the highest '
            'mean-rating-to-mean-price ratio (excluding free apps)?'
        )
        kids_df = df[
            ((df["name"].str.lower().str.contains("kids", na=False))  # name contains "kids"
             | (df["app_desc"].str.lower().str.contains("kids", na=False)))  # description contains "kids"
            & (df["price"] > 0)  # paid apps
        ].copy()
        grouped = kids_df.groupby("prime_genre").agg(  # group by category
            avg_rating=("user_rating", "mean"),  # average rating
            avg_price=("price", "mean")  # average price
        ).reset_index()
        grouped["ratio"] = grouped["avg_rating"] / \
            grouped["avg_price"]  # mean-rating-to-mean-price ratio
        sorted_df = grouped.sort_values(
            by="ratio", ascending=False)  # sort by ratio
        ground_truth = sorted_df.iloc[0]["prime_genre"] if len(
            sorted_df) > 0 else "None"

        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Very Hard",
            tokens=tokens_in_step
        )

    def _veryhard_stddev_top5_cat_above_0_99(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = (
            "Among the top 5 categories by total reviews, which has the largest rating standard deviation "
            "for apps priced above $0.99?"
        )
        # top 5 categories by total reviews
        cat_reviews = df.groupby("prime_genre")[
            "rating_count_tot"].sum().reset_index()
        # sort by total reviews
        cat_reviews_sorted = cat_reviews.sort_values(
            by="rating_count_tot", ascending=False).head(5)
        top_5_cats = cat_reviews_sorted["prime_genre"].tolist()

        # filter for apps in top 5 categories and priced above $0.99
        filtered = df[df["prime_genre"].isin(
            top_5_cats) & (df["price"] > 0.99)].copy()
        # group by category and calculate standard deviation of ratings
        grouped = filtered.groupby("prime_genre")[
            "user_rating"].std().reset_index()
        # sort by standard deviation
        sorted_std = grouped.sort_values(by="user_rating", ascending=False)
        if len(sorted_std) > 0:
            ground_truth = sorted_std.iloc[0]["prime_genre"]
        else:
            ground_truth = "None"

        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Very Hard",
            tokens=tokens_in_step
        )

    def _veryhard_3_apps_mention_two_of_learning_assistant_ai_study(self, df: pd.DataFrame, context_str: str, tokens_in_step: int) -> Example:
        question = (
            'What are the top 3 apps by reviews-to-rating ratio that explicitly mention at least two of the words: "learning", '
            '"assistant", "AI", "study" and have 10,000+ reviews?'
        )

        keywords = ["learning", "assistant", "ai", "study"]

        def count_matches(text: str) -> int:
            text_lower = text.lower()
            return sum(kw in text_lower for kw in keywords)

        df_copy = df.copy()
        # Add a column for how many keywords each row matches
        df_copy["matches_count"] = df_copy.apply(
            lambda row: count_matches(
                (row["name"] or "") + " " + (row["app_desc"] or "")),
            axis=1
        )

        filtered = df_copy[(df_copy["matches_count"] >= 2)
                           & (df_copy["rating_count_tot"] >= 10000)]
        # Avoid dividing by zero if rating is 0
        safe_filtered = filtered[filtered["user_rating"] > 0].copy()
        safe_filtered["reviews_to_rating"] = safe_filtered["rating_count_tot"] / \
            safe_filtered["user_rating"]

        sorted_df = safe_filtered.sort_values(
            by="reviews_to_rating", ascending=False).head(3)
        ground_truth = get_app_names(sorted_df)
        return Example.create(
            question=question,
            ground_truth=ground_truth,
            difficulty="Very Hard",
            tokens=tokens_in_step
        )


if __name__ == "__main__":
    gt_gen = GroundTruthGenerator()
    gt_gen.generate_examples()
