import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_eval_dataset():
    """Load the evaluation dataset."""
    with open("output/eval_dataset.json", "r") as f:
        return json.load(f)


def run_llm_eval(prompt):
    """Run evaluation using OpenAI API."""
    client = OpenAI()
    pass


def compare_with_ground_truth(llm_response, ground_truth):
    """Compare LLM response with ground truth."""
    pass


def log_results(results):
    """Log evaluation results."""
    pass


def main():
    # Load evaluation dataset
    eval_dataset = load_eval_dataset()

    results = []
    for example in eval_dataset:
        llm_response = run_llm_eval(example["prompt"])
        accuracy = compare_with_ground_truth(
            llm_response, example["ground_truth"])

        results.append({
            "question": example["question"],
            "context_size": example["context_size"],
            "accuracy": accuracy
        })

    # Log results
    log_results(results)

    # Save results
    with open("output/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
