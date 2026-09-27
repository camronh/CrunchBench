# CrunchBench

CrunchBench asks LLMs analytical questions like **"Among paid apps, which 3 have the highest ratio of rating count to price?"** about a pile of raw [App Store listings](https://www.kaggle.com/datasets/ramamet4/app-store-apple-data-set-10k-apps). The same 20 questions are asked with 8k to 512k tokens of app data, doubling at each step, so you can see how well a model filters, sorts, counts and aggregates as the context grows.

It tests a model's ability to:

- Find the relevant rows in long, unstructured context
- Apply several filters and tie-breakers exactly
- Do multi-step arithmetic over groups of rows

The benchmark runs on [EZVals](https://ezvals.com). Models are called through [OpenRouter](https://openrouter.ai) with their native reasoning turned on, and every call is traced to LangSmith with the trace linked from each EZVals result.

## Dataset

Each app is shown to the model like this:

```markdown
App Name: Google Search made just for mobile
ID: 284815942
Size: 217.24 MB
Price: 0.0 USD
Rating Count: 479440
Average User Rating: 3.5
Version: 27.0
Category: Utilities
Description: The Google app keeps you in the know about the things you care about. ...
```

The data is the 1,731 most-reviewed apps that fit in 512k tokens, shown in ID order. Each smaller context size is the first part of the same list:

| Context size | 8k | 16k | 32k | 64k | 128k | 256k | 512k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Apps | 26 | 53 | 107 | 215 | 421 | 840 | 1,731 |

Token counts use OpenAI's cl100k tokenizer. Other tokenizers count differently, and Claude's counts about 1.6 times as many.

Descriptions longer than 1,200 characters are cut at the last line or sentence break and end with " [...]". The tail of a long listing is mostly legal notices, subscription terms and press quotes. Cutting it keeps every app while capping the noise. Ground truth is computed on the shortened text, so it always matches what the model sees.

There are 5 questions at each of 4 difficulty levels. Every question is asked at all 7 context sizes, for 140 cases per model. See [questions.md](./questions.md) for every question and its answer at every context size.

| Difficulty | Example |
| --- | --- |
| Easy | In the 'Games' category, which 5 apps have the highest rating count? |
| Medium | Among paid apps, which 5 have the lowest rating? Break ties by lower rating count. |
| Hard | What is the mean price of the paid apps in the 'Games' category? Round to 2 decimal places. |
| Very Hard | Among apps whose rating is strictly higher than the mean rating of all apps in their own category, which 3 have the highest rating count? |

### Ground truth

Answers are computed with pandas in [crunchbench/questions.py](./crunchbench/questions.py) over exactly the apps in each context. Every computation also checks that its answer is unambiguous: no ties at a ranking cutoff, no values on a rounding edge, and no keyword whose matches change between whole-word, substring and plural readings. [scripts/build_dataset.py](./scripts/build_dataset.py) refuses to write the dataset if any check fails.

[tests/test_ground_truth.py](./tests/test_ground_truth.py) then re-derives all 100 answers independently. It parses the exact text the model receives and recomputes each answer in plain Python with exact fractions, without pandas.

Every question is sent with a short set of rules that pin down terms like "rating count", "free" and "contains the word". They're listed at the top of [questions.md](./questions.md).

### Scoring

Models answer in JSON, and [crunchbench/grading.py](./crunchbench/grading.py) grades each answer in code. There's no LLM judge. Each question expects one of four answer types:

| Type | Model returns | Graded on |
| --- | --- | --- |
| Apps | `{"apps": [{"id": ..., "name": ...}]}` | App IDs, which are unique |
| Categories | `{"categories": [...]}` | Category names, ignoring case and spacing |
| Count | `{"count": ...}` | Exact integer |
| Number | `{"value": ...}` | Value at the requested precision |

Scores:

- **Correct (1.0):** exactly the right items, in the right order when order matters, or the right number.
- **Partially correct (0.5):** the answer has more than one right item, and includes some of them but misses some, adds others or has them out of order.
- **Incorrect (0.0):** none of the right items, a wrong count or number, or output that doesn't match the schema.

A result counts as passed in EZVals only when it's fully correct. The partial-credit score is stored as the score's value. Output that doesn't parse is scored 0 and flagged with `format_ok: false`, so format failures stay separate from wrong answers.

Models that OpenRouter lists as supporting structured outputs get a strict JSON schema, routed only to providers that enforce it. Other models get the schema in the prompt, and the grader reads the JSON from their reply.

## Setup

```bash
uv sync
cp .env.example .env   # then add your OpenRouter and LangSmith keys
```

## Running

Each model is an EZVals config profile in [ezvals.json](./ezvals.json). Pick one with `--config`, and the profile name becomes the run name:

```bash
# One model, all 140 cases, one at a time so the prompt cache is reused
uv run ezvals run evals/ --config gpt-6-sol --session v3 -c 1

# A quick check on a few cases
uv run ezvals run evals/ --config gpt-6-luna --session smoke --limit 3

# One case
uv run ezvals run "evals/benchmark.py::crunchbench@V5-64k" --config claude-opus-5.5 --session smoke

# Only some difficulties or context sizes (labels match any: easy, medium, hard, very-hard, 8k ... 512k)
uv run ezvals run evals/ --config gpt-6-sol --session v3 --label very-hard

# Every profile, then the report
scripts/run_all.sh v3

# Browse, rerun and compare runs in the web UI
uv run ezvals serve evals/ --session v3
```

`ezvals serve` can compare runs side by side. Each result links to its full LangSmith trace, and stores cost, token counts, the provider that served it and the model's reasoning summary.

Runs are never overwritten. To retry a few failed cases, run them again in the same session. The report uses the latest result for each case.

**Concurrency and cost.** Questions at the same context size share the app data, so after the first call the rest can read it from the prompt cache at about a tenth of the price. Calls that run in parallel all miss the cache and pay a 1.25x write premium instead. Run expensive models with `-c 1`. `scripts/run_all.sh` uses each profile's `concurrency` value.

### Adding or changing models

Add a profile under `configs` in [ezvals.json](./ezvals.json). A profile needs an OpenRouter model id and can override any setting in `crunchbench.model_defaults`:

```json
"gemini-3.8-flash": { "model": "google/gemini-3.8-flash", "reasoning_effort": "medium" }
```

| Setting | Default | What it does |
| --- | --- | --- |
| `reasoning_effort` | `high` | OpenRouter reasoning effort. Use `null` to turn reasoning off. |
| `max_tokens` | `null` | Output budget, including reasoning tokens. `null` uses the model's own maximum on the providers the request can reach, so a cut-off answer means the model couldn't finish, not that the benchmark stopped it. |
| `temperature` | `null` | `null` uses the provider default |
| `prompt_cache` | `true` | Marks the end of the app data as a cache breakpoint for Anthropic and OpenAI models. Other providers cache shared prefixes automatically. |
| `structured_output` | `auto` | `auto` uses a strict schema when the model supports it. `strict` or `prompt` forces one mode. |
| `provider` | none | [OpenRouter provider routing](https://openrouter.ai/docs/features/provider-routing), such as `{"only": ["DeepInfra"], "allow_fallbacks": false}` |
| `concurrency` | `1` | Cases at once when run through `scripts/run_all.sh` |

Pin open-weight models to one provider. OpenRouter serves them from many hosts, some with more heavily compressed weights, and results can differ between hosts. DeepSeek is pinned to DeepInfra, which publishes fp8 weights and supports strict schemas. DeepSeek's own endpoint only works if your OpenRouter privacy settings allow providers that train on prompts. It also lacks strict schemas, so its profile would need `"structured_output": "prompt"`.

A model that runs out of output budget gets no credit, and the report counts those answers in a separate "Cut off" column. Because the default budget is each model's own maximum, a cut-off means the model couldn't finish. On OpenRouter, Anthropic models size their thinking allowance from the output budget, so a larger budget also gives Claude more room to think, and can raise its cost.

The context sizes are set under `crunchbench` in the same file. After changing them, rerun `scripts/preprocess.py` and `scripts/build_dataset.py`, since the app pool is sized to the largest context. The per-eval timeout is an ordinary EZVals setting at the top level.

Check long-context pricing before large runs. Some models charge more past a prompt-size threshold. GPT-6 Luna, for example, doubles its input price above 272k tokens, which applies to the 512k size.

### Report

```bash
uv run scripts/report.py --session v3
```

This writes the charts in `data/plots/` and a results table in `data/results.md`, using the latest result for each profile and case in the session. The context-size chart uses a log axis, so each doubling gets the same width.

## Maintaining the dataset

```bash
uv run scripts/preprocess.py     # raw Kaggle CSVs in data/ -> data/appstore_preprocessed.csv
uv run scripts/build_dataset.py  # questions -> data/eval_dataset.json and questions.md
uv run pytest                    # independent check of every ground truth
```

## Version 1 results

The January 2025 results are kept in [data/v1](./data/v1) for reference. They aren't comparable with this version. Version 1 had ground-truth problems that this version fixes:

- Most rankings were decided by ties, because ratings come in half-star steps and no tie-breaker was given.
- Keyword checks matched substrings, so "AI" matched words like "again".
- Several questions had empty answers at every context size, and one skipped a filter its text asked for.
- Every app's size was shown to the model as "10 MB".
