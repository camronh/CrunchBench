from utils import load_data, slice_dataframe_by_tokens, store_results, dataset_name
from langsmith import evaluate, Client
from models import Example
from langchain_openai import ChatOpenAI
import os
from judges import correctness_judge
import time

class Model:
    def __init__(self, name: str):
        self.name = name
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
            model=self.name,
        )
        self.df = load_data()

    def target(self, input: Example) -> str:
        """
        Target function for the evals
        """
        # Get the df slice
        app_str, _ = slice_dataframe_by_tokens(self.df, input["tokens"])

        prompt = f"""<APP STORE DATA>
{app_str}
</APP STORE DATA>

You are tasked with analyzing the <APP STORE DATA> to answer the <QUESTION> below as accurately and as succinctly as possible. Feel \
free to BRIEFLY reason through your decision first to make sure you get your <ANSWER right, then finally wrap your answer in <ANSWER> tags:

<QUESTION>
{input["question"]}
</QUESTION>"""

        return {"answer": self._invoke(prompt)}

    def _invoke(self, prompt: str) -> str:
        """
        Invoke a model and return the response string with one retry attempt
        """
        for attempt in range(2):  # Will try twice: initial attempt + 1 retry
            try:
                return self.llm.invoke(prompt).content
            except Exception as e:
                if attempt == 0:  # Only print and continue if it's the first attempt
                    print(f"First attempt failed: {e}. Retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                print(f"Retry failed: {e}")
                raise e  # Re-raise the exception after retry fails


def run_evals(model_name: str, custom_label: str = None, max_concurrency: int = 5):
    """
    Run CrunchBench evals for a given model
    """
    model = Model(model_name)
    client = Client()

    # Run evals
    results = evaluate(
        model.target,
        data=client.list_examples(dataset_name=dataset_name, splits=[
                                  "10k Tokens", "20k Tokens", "30k Tokens", "40k Tokens", "50k Tokens"]),
        evaluators=[correctness_judge],
        max_concurrency=max_concurrency,
        experiment_prefix=custom_label or model_name,
    )

    store_results(results, model_name, custom_label)
    return results
