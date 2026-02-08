#!/usr/bin/env python3
"""
Plot TTT Dreamer metrics from one or more runs.

Usage:
  python plot_ttt.py outputs/ttt_20260207_143000/
  python plot_ttt.py outputs/ttt_*/                   # overlay multiple runs
  python plot_ttt.py outputs/ttt_*/ --save plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"

    rows = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                # Handle "inf" perplexity
                if row.get("perplexity") == "inf":
                    row["perplexity"] = float("nan")
                rows.append(row)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    label = f"lr={config.get('lr', '?')}"
    if config.get("control"):
        label = "control"

    return {"rows": rows, "config": config, "label": label, "dir": run_dir.name}


def plot_runs(runs, save_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("TTT Dreamer — Self-Training Collapse Dynamics", fontsize=14, fontweight="bold")

    fields = [
        ("loss", "Loss (CE)"),
        ("perplexity", "Perplexity"),
        ("distinct_1", "Distinct-1 (unigrams)"),
        ("distinct_2", "Distinct-2 (bigrams)"),
        ("token_entropy", "Token Entropy"),
        ("top1_prob", "Top-1 Probability"),
        ("weight_drift", "Weight Drift (L2)"),
        ("grad_norm", "Gradient Norm"),
    ]

    for idx, (field, title) in enumerate(fields):
        ax = axes[idx // 3][idx % 3]
        for run in runs:
            steps = [r["step"] for r in run["rows"] if field in r]
            vals = [r[field] for r in run["rows"] if field in r]
            # Convert to float, skip NaN for plotting
            vals = [float(v) if v is not None else float("nan") for v in vals]
            ax.plot(steps, vals, label=run["label"], alpha=0.8, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=7)

    # Last subplot: text snippets from first run
    ax = axes[2][2]
    ax.axis("off")
    if runs:
        run = runs[0]
        snippets = []
        sample_steps = np.linspace(0, len(run["rows"]) - 1, min(6, len(run["rows"])), dtype=int)
        for i in sample_steps:
            row = run["rows"][i]
            snip = row.get("text_snippet", "")[:80].replace("\n", " ")
            snippets.append(f"step {row['step']}: {snip!r}")
        ax.text(0.05, 0.95, "\n\n".join(snippets), transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title("Text Samples")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TTT Dreamer metrics")
    parser.add_argument("runs", nargs="+", help="Run directories (outputs/ttt_*/)")
    parser.add_argument("--save", type=str, default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    runs = []
    for path in args.runs:
        p = Path(path)
        if (p / "metrics.jsonl").exists():
            runs.append(load_run(p))
        else:
            print(f"Skipping {path}: no metrics.jsonl found")

    if not runs:
        print("No valid runs found.")
        return

    print(f"Plotting {len(runs)} run(s): {[r['label'] for r in runs]}")
    plot_runs(runs, save_path=args.save)


if __name__ == "__main__":
    main()
