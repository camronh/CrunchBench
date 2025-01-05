# LLM Analytics Benchmark

Welcome to the LLM Analytics Benchmark! This project tests how well large language models can handle structured data queries—especially when the dataset is large enough to bust the typical context window. We’re talking partial context, filtering, sorting, and all the tough stuff LLMs usually struggle with.

## What’s This All About?

We’re taking a big dataset of App Store data, trimming it down to about 150k tokens, and then challenging LLMs to answer tricky queries about that data—like “Which 3 apps have the highest number of reviews but a rating under 4.5?” We slice the dataset into multiple context sizes (like 10k tokens, 20k tokens, and so on) and watch the model sweat when it only gets partial information.

## Quick Start

1. **Clone the repo**

   ```
   git clone https://github.com/yourusername/llm-analytics-benchmark.git
   cd llm-analytics-benchmark
   ```

2. **Install dependencies**  
   We use Poetry for dependency management, but you can also use pip with the exported requirements.

   With Poetry (recommended):

   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install
   ```

   With pip:

   ```bash
   # Export requirements and install
   poetry export -f requirements.txt --output requirements.txt --without-hashes
   pip install -r requirements.txt
   ```

3. **Dataset Setup**

   - Place the App Store dataset files (`AppleStore.csv` and `appleStore_description.csv`) in the `data/` directory
   - Run the preprocessing script to prepare the data for evaluation

4. **Generate the evaluation dataset**

   - This step creates all your partial contexts and ground-truth answers.

5. **Run the evals**
   - Finally, feed everything to your favorite LLM and see how it performs.

That's it in a nutshell. If you want more details, read on.

## Files and Folders

- **data/**  
  Where raw and preprocessed CSVs live (like `appstore_preprocessed.csv`).

- **scripts/**  
  Where your main Python scripts go. Think:

  1. `01_download_and_preprocess.py` – Grabs the Kaggle dataset, cleans it, and truncates to ~120k tokens.
  2. `02_generate_eval_dataset.py` – Generates partial contexts for each context length, plus ground-truth answers.
  3. `03_run_evals.py` – Calls the LLM on each question/context pair and logs how well it does.

- **config/**  
  Might hold things like `questions.json`, where each question is described (filters, sorting, etc.).

- **output/**  
  Houses generated files (partial context text, ground-truth JSON, etc.).

- **README.md** (This file!)  
  Explains what this project is and how to replicate it.

## Step-by-Step Flow

1. **Preprocess**

   ```
   poetry run python scripts/01_download_and_preprocess.py
   ```

   - Takes the App Store data files from the `data/` directory
   - Cleans it, assigns stable IDs, and stops once we reach ~120k tokens total
   - Saves `appstore_preprocessed.csv` to `data/`

2. **Generate the Evaluation Dataset**

   ```
   python scripts/02_generate_eval_dataset.py
   ```

   - Reads `appstore_preprocessed.csv`.
   - Loads your questions (like "top 3 apps under rating 4.5?").
   - For each question, figures out the ground-truth IDs from the full dataset.
   - Then for each context size (10k, 20k, …, up to 120k tokens), creates a "partial slice" of the dataset and builds the prompt.
   - Saves a giant JSON (or multiple files) with everything needed to run the tests.

3. **Run the Evals**
   ```
   python scripts/03_run_evals.py
   ```
   - Reads the big JSON of question/context pairs.
   - Calls your chosen LLM for each example.
   - Compares the LLM's answer to the ground truth (IDs, ranks, etc.).
   - Logs the success rate. This can also be integrated with LangSmith or any monitoring platform you like.

## Token Counting

At multiple points we talk about "10k tokens," "20k tokens," etc. We approximate how many tokens each row in the dataset will add to the final prompt. Typically, you'll do something like:

- Combine the relevant text fields into one string (e.g., app name, category, description).
- Use a tokenizer (like the one from `tiktoken` if you're using OpenAI) to count tokens.
- Keep adding rows until you reach your target.

This ensures that when you say "10k tokens of context," it's roughly accurate. The details are up to you.

## How to Contribute or Reproduce

1. **Fork or clone** this repo.
2. Make sure you install the required Python packages with `pip install -r requirements.txt`.
3. Run each script in order—anyone should get the same final dataset and evaluation results if they follow these steps.
4. If you want to push your results to LangSmith, set your environment variables or tweak the code in `03_run_evals.py`.
