"""Summarize an EZVals session into charts and a Markdown results table.

    uv run scripts/report.py --session v3

Reads every run in .ezvals/sessions/<session>/, keeps the latest result for each
model profile and case, and writes data/plots/*.png plus data/results.md (next to the plots folder).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crunchbench.data import DATA_DIR, ROOT
from crunchbench.dataset import load_dataset

# Validated categorical palette (fixed order, never cycled) and neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIFFICULTIES = ["Easy", "Medium", "Hard", "Very Hard"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def load_session(session: str) -> pd.DataFrame:
    folder = ROOT / ".ezvals" / "sessions" / session
    runs = sorted(folder.glob("*.json"), key=lambda p: json.loads(p.read_text()).get("created_at", 0))
    if not runs:
        raise SystemExit(f"No runs found in {folder}")

    # Runs are never overwritten, so a profile can have a full run plus reruns of
    # a few cases. Keep the latest result for each (profile, question, size).
    latest: dict[tuple, dict] = {}
    for path in runs:
        run = json.loads(path.read_text())
        name = run.get("config_name") or run.get("run_name") or path.stem
        for item in run["results"]:
            result = item["result"]
            meta = result.get("metadata") or {}
            if meta.get("question_id") is None or result.get("status") == "not_started":
                continue
            latest[(name, meta["question_id"], meta.get("tokens"))] = result

    rows = []
    for (name, question_id, tokens), result in latest.items():
        meta = result.get("metadata") or {}
        score = next((s for s in result.get("scores") or [] if s.get("key") == "correctness"), None)
        rows.append(
            {
                "model": name,
                "question_id": question_id,
                "difficulty": meta.get("difficulty"),
                "tokens": tokens,
                # Errors count as zero so a model can't gain by failing to answer.
                "score": (score or {}).get("value") or 0.0,
                "strict": 1.0 if (score or {}).get("passed") else 0.0,
                "error": bool(result.get("error")),
                "format_fail": meta.get("format_ok") is False,
                "cut_off": bool(meta.get("cut_off")) or meta.get("finish_reason") == "length",
                "cost": meta.get("cost_usd") or 0.0,
                "latency": result.get("latency") or 0.0,
            }
        )
    return pd.DataFrame(rows)


def colors_for(models: list[str]) -> dict[str, str]:
    if len(models) > len(SERIES):
        raise SystemExit(f"Too many models ({len(models)}) for the {len(SERIES)}-color palette; split the report.")
    return {m: SERIES[i] for i, m in enumerate(models)}


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overall(summary: pd.DataFrame, out: Path) -> None:
    s = summary.sort_values("score")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.4))
    y = range(len(s))
    ax.barh(y, s["strict"] * 100, color=SERIES[0], height=0.6, label="Fully correct")
    ax.barh(
        y,
        (s["score"] - s["strict"]) * 100,
        left=s["strict"] * 100,
        color="#a9c9ef",
        height=0.6,
        label="Partial credit",
        edgecolor=SURFACE,
        linewidth=2,
    )
    for i, v in enumerate(s["score"] * 100):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", color=TEXT)
    ax.set_yticks(list(y), s.index)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    ax.set_title("Average score", loc="left", fontweight="bold", pad=28)
    save(fig, out, "1_average_scores.png")


def plot_lines(
    df: pd.DataFrame,
    by: str,
    order: list,
    labels: list[str],
    colors: dict,
    out: Path,
    name: str,
    title: str,
    log_x: bool = False,
    xlabel: str | None = None,
):
    """One line per model across `order`.

    With `log_x`, points sit at their numeric values on a base-2 log axis, so each
    doubling gets equal width. Otherwise they are evenly spaced categories.
    """
    pivot = df.pivot_table(index=by, columns="model", values="score", aggfunc="mean").reindex(order) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [float(v) for v in order] if log_x else list(range(len(order)))
    for model, color in colors.items():
        if model not in pivot:
            continue
        ax.plot(x, pivot[model], color=color, linewidth=2, marker="o", markersize=6, label=model)
        points = pivot[model].dropna()
        if len(colors) <= 4 and not points.empty:  # Direct labels only while they stay legible.
            ax.annotate(
                model,
                (x[order.index(points.index[-1])], points.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=TEXT,
                fontsize=9,
            )
    if log_x:
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
    ax.set_xticks(x, labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    save(fig, out, name)


def plot_cost(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(summary["cost"], summary["score"] * 100, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for model, row in summary.iterrows():
        ax.annotate(model, (row["cost"], row["score"] * 100), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Cost of one full run (USD, log scale)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Score vs. cost", loc="left", fontweight="bold")
    save(fig, out, "4_price_performance.png")


def markdown(summary: pd.DataFrame, df: pd.DataFrame) -> str:
    diff = df.pivot_table(index="model", columns="difficulty", values="score", aggfunc="mean")
    lines = [
        "| Model | Score | Fully correct | "
        + " | ".join(DIFFICULTIES)
        + " | Cost | Cut off | Format failures | Errors |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in DIFFICULTIES) + " | ---: | ---: | ---: | ---: |",
    ]
    for model, r in summary.sort_values("score", ascending=False).iterrows():
        by_diff = " | ".join(
            f"{diff.loc[model, d] * 100:.1f}%" if d in diff and pd.notna(diff.loc[model, d]) else "-"
            for d in DIFFICULTIES
        )
        lines.append(
            f"| {model} | {r['score'] * 100:.1f}% | {r['strict'] * 100:.1f}% | {by_diff} | "
            f"${r['cost']:.2f} | {int(r['cut_offs'])} | {int(r['format_fails'])} | {int(r['errors'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="EZVals session name, e.g. v3")
    parser.add_argument("--out", default=str(DATA_DIR / "plots"), help="Folder for the PNG charts")
    args = parser.parse_args()

    df = load_session(args.session)
    summary = df.groupby("model").agg(
        score=("score", "mean"),
        strict=("strict", "mean"),
        cost=("cost", "sum"),
        errors=("error", "sum"),
        format_fails=("format_fail", "sum"),
        cut_offs=("cut_off", "sum"),
        n=("score", "size"),
    )
    expected = len(load_dataset()["cases"])
    incomplete = summary[summary["n"] < expected]
    if not incomplete.empty:
        print(f"Warning: incomplete models (of {expected} cases): {incomplete['n'].to_dict()}")
    errored = summary[summary["errors"] > 0]
    if not errored.empty:
        print(
            f"Warning: errors are scored as 0: {errored['errors'].astype(int).to_dict()}. "
            "Rerun them with `ezvals serve` or a case selector before publishing."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    colors = colors_for(list(summary.sort_values("score", ascending=False).index))

    plot_overall(summary, out)
    plot_lines(
        df, "difficulty", DIFFICULTIES, DIFFICULTIES, colors, out, "2_difficulty_breakdown.png", "Score by difficulty"
    )
    sizes = sorted(df["tokens"].dropna().unique())
    plot_lines(
        df,
        "tokens",
        sizes,
        [f"{int(t) // 1000}k" for t in sizes],
        colors,
        out,
        "3_token_size_breakdown.png",
        "Score by context size",
        log_x=True,
        xlabel="Context size in tokens (each step doubles)",
    )
    plot_cost(summary, out)

    table = markdown(summary, df)
    (out.parent / "results.md").write_text(f"# CrunchBench results (session `{args.session}`)\n\n{table}")
    print(table)
    print(f"Charts written to {out}")


if __name__ == "__main__":
    main()
