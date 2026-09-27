# CrunchBench v2 handoff

Status as of 2026-09-27. Written to carry the work over to another machine.

## Set up on a new machine

```bash
git clone https://github.com/camronh/CrunchBench.git
cd CrunchBench
git checkout ezvals-v2
uv sync
cp .env.example .env   # then fill in the values below
uv run pytest          # 179 tests; checks every ground truth independently
```

`.env` needs these variables. The values are not in the repo.

| Variable | What it is |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter inference key named "CrunchBench", with a $20 limit. About $17.38 was left on 2026-09-27. |
| `OPENROUTER_MANAGEMENT_KEY` | OpenRouter management key. It can create and inspect keys but can't call models. |
| `LANGSMITH_API_KEY` | LangSmith key, used only for traces. Each EZVals result links to its trace. |
| `LANGSMITH_TRACING` | `true` |
| `LANGSMITH_PROJECT` | `CrunchBench` |

The management key was pasted into a chat on 2026-09-24, so rotating it is a good idea.

## What changed from v1

- The eval runs on EZVals (`evals/benchmark.py`). Each model is a config profile in `ezvals.json`, and the profile name becomes the run name.
- The LangChain and LangSmith evaluation code, the notebook and Poetry are gone. The project uses uv.
- There's no LLM judge. Models answer in a strict JSON schema, and `crunchbench/grading.py` grades in code: 1 for exact, 0.5 for partial list credit, 0 otherwise.
- The questions were rebuilt. v1 ground truths were mostly decided by ties, substring keyword matches and a bug that showed every app's size as "10 MB". v1 results are kept in `data/v1/` for reference only.
- Every ground truth is checked for ties, rounding edges and ambiguous keywords when the dataset is built. `tests/test_ground_truth.py` re-derives all 140 answers from the exact prompt text without pandas.
- Context sizes double from 8k to 512k (cl100k tokens), with 26 to 1,731 apps. The chart uses a log axis.
- Long descriptions are cut at about 1,200 characters at a clean break, not dropped.
- The output budget defaults to each model's own maximum on the providers it can reach, looked up from OpenRouter. An answer cut off at that limit is a legitimate failure and is reported in its own column.
- Prompt caching uses an explicit breakpoint on the app data. Cases run grouped by context size and answer type.
- Runs are never overwritten. The report uses the latest result per profile and case.

## Results so far

GPT-6 Luna is the only model run on the final setup: session `v3`, 140 cases.

| Context | 8k | 16k | 32k | 64k | 128k | 256k | 512k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Score | 97.5% | 92.5% | 72.5% | 82.5% | 75.0% | 72.5% | 55.0% |

The overall score is 78.2%, with 96 of 140 fully correct. The run cost about $1.29, with no errors and no cut-offs. The count questions and the mean-price question fail from 32k up. At 512k, Luna consistently undercounts, answering 347, 561, 575 and 769 when the truth is 828.

- Run files: `.ezvals/sessions/v3/`. The `luna-retry-*` runs are repeat attempts of the two 512k cases that were cut off at the old 32k budget.
- Report: `data/results.md` and `data/plots/`. Rebuild with `uv run scripts/report.py --session v3`.
- Chart page: `reports/luna-v3.html`, also published privately at https://claude.ai/artifact/QTLdduffcF4oKyZcNzp7yS
- `.ezvals/sessions/v2/` is an earlier 10k to 50k pilot on the old data. It's not comparable with v3.

## Decisions made

- No LLM judge, no F1, no confidence intervals or repeated runs.
- No closed-book baseline, no perturbed data, no templated questions, no shuffled app order, no tool-use ceiling.
- Log-scale doubling ladder instead of linear steps.
- Keep noisy long descriptions, cut short at about 1,200 characters.
- DeepSeek is pinned to DeepInfra (fp8, strict schemas). DeepSeek's own endpoint is blocked by the account's OpenRouter privacy setting on providers that train on prompts.

## Next steps

1. Add operation tags to each question (lookup, filter, sort, count, aggregate, group-by) as EZVals labels, and report scores per tag.
2. Show apps per context alongside tokens on the context-size chart.
3. Run the other profiles: `claude-opus-5.5`, `gpt-6-sol` and `deepseek-v4.1-flash` on session `v3`. Use each profile's concurrency: `scripts/run_all.sh v3` runs all of them and builds the report. Opus at 512k likely needs more credit than the key has left, so consider more credit or a first pass capped at 256k.
4. Optional: a reasoning-effort axis for one or two models (low and medium), and a 1M step for models whose windows allow it.
5. Before publishing: a private test split with a canary string, public traces, and a short write-up that covers method, audit, error analysis, related work (RULER, NoLiMa, LongBench v2, HELMET, and aggregation-focused work like Oolong) and limitations. The main limitation is 20 questions per size and a single run.

## Useful commands

```bash
uv run ezvals run evals/ --config gpt-6-luna --session v3 -c 8                  # one model
uv run ezvals run "evals/benchmark.py::crunchbench@M3-512k" --config gpt-6-luna --session v3   # one case
uv run ezvals serve evals/ --session v3 --config gpt-6-luna --run-name retry   # UI, reruns
uv run scripts/preprocess.py && uv run scripts/build_dataset.py && uv run pytest   # after data or question changes
```

If the server says a run name is ambiguous, pass a new `--run-name`. The profile still sets the model.
