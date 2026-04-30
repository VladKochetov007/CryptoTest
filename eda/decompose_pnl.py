"""Decision-vs-execution PnL decomposition.

Reads trade logs from eda/backtest/trades_{universe}.csv (written by backtest.py)
and decomposes per-trade PnL into three orthogonal components:

  gross_roi     = exit/entry - 1                         (price-only, no friction)
  fee_drag      = 2 * TAKER_FEE = 0.02                  (flat protocol cost)
  slip_drag     = AMM price impact net of flat fee       (size-dependent friction)
  net_roi       = gross_roi - fee_drag - slip_drag

Decision score (model probability) vs actual label tells us:
  - Alpha capture: does higher score predict higher gross_roi?
  - Missed alpha: what fraction of available gross gain did the exit capture?
    capture_rate = gross_roi / max_possible_roi  (max = peak_price/entry - 1)

Outputs:
  eda/plots/decompose_alpha_capture.png   — scatter decision_score vs gross_roi
  eda/plots/decompose_friction.png        — fee vs slip contribution by strategy
  eda/plots/decompose_capture_rate.png    — CDF of alpha capture rate by strategy
  eda/backtest/decomposition_{universe}.json  — summary table
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "eda" / "backtest"
PLOT_DIR = ROOT / "eda" / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_trades(universe: str) -> pl.DataFrame:
    path = BACKTEST / f"trades_{universe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"trade log not found: {path}. Run backtest.py first.")
    df = pl.read_csv(path)
    # keep rows where we have a valid decision_score and actual label
    return df.with_columns([
        pl.col("decision_score").cast(pl.Float64),
        pl.col("gross_roi").cast(pl.Float64),
        pl.col("net_roi").cast(pl.Float64),
        pl.col("fee_drag").cast(pl.Float64),
        pl.col("slip_drag").cast(pl.Float64),
        pl.col("peak_price").cast(pl.Float64),
        pl.col("entry_price").cast(pl.Float64),
        pl.col("hit_2x_actual").cast(pl.Int32),
    ])


def alpha_capture_rate(df: pl.DataFrame) -> pl.DataFrame:
    """Max achievable gross ROI = peak_price/entry - 1. Capture = actual/max."""
    max_roi = df["peak_price"] / df["entry_price"] - 1.0
    captured = df["gross_roi"] / max_roi.clip(lower_bound=1e-6)
    return df.with_columns([
        max_roi.alias("max_achievable_roi"),
        captured.clip(upper_bound=1.0).alias("alpha_capture_rate"),
    ])


def plot_alpha_capture(df: pl.DataFrame, universe: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: decision_score vs gross_roi scatter (sample 3k)
    scored = df.filter(pl.col("decision_score").is_not_nan())
    if scored.height > 3000:
        scored = scored.sample(3000, seed=42)
    ax = axes[0]
    ax.scatter(scored["decision_score"].to_numpy(),
               scored["gross_roi"].to_numpy().clip(-1, 5),
               alpha=0.15, s=6, c="steelblue")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("decision_score (model prob)")
    ax.set_ylabel("gross_roi (price-only)")
    ax.set_title(f"Model score vs price return [{universe}]")

    # right: alpha_capture_rate CDF per strategy
    ax2 = axes[1]
    for strat in df["strategy"].unique().to_list():
        sub = df.filter(pl.col("strategy") == strat)
        sub = alpha_capture_rate(sub).filter(pl.col("alpha_capture_rate").is_not_nan())
        cap = np.sort(sub["alpha_capture_rate"].to_numpy())
        ax2.plot(cap, np.linspace(0, 1, len(cap)), label=strat, linewidth=1.2)
    ax2.set_xlabel("alpha capture rate (actual/max)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Alpha capture CDF by strategy")
    ax2.legend(fontsize=7)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = PLOT_DIR / "decompose_alpha_capture.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_friction(df: pl.DataFrame, universe: str):
    strategies = df["strategy"].unique().to_list()
    mean_fee = []
    mean_slip = []
    for s in strategies:
        sub = df.filter(pl.col("strategy") == s)
        mean_fee.append(float(sub["fee_drag"].mean()))
        mean_slip.append(float(sub["slip_drag"].mean()))

    x = np.arange(len(strategies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mean_fee, width, label="fee_drag", color="#e74c3c")
    ax.bar(x + width / 2, mean_slip, width, label="slip_drag", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean friction (fraction of capital)")
    ax.set_title(f"Fee vs AMM slip drag by strategy [{universe}]")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = PLOT_DIR / "decompose_friction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def summary_table(df: pl.DataFrame) -> list[dict]:
    rows = []
    for strat in sorted(df["strategy"].unique().to_list()):
        sub = df.filter(pl.col("strategy") == strat)
        sub = alpha_capture_rate(sub)
        valid_cap = sub.filter(pl.col("alpha_capture_rate").is_not_nan())
        scored = sub.filter(pl.col("decision_score").is_not_nan())
        rows.append({
            "strategy": strat,
            "n_trades": sub.height,
            "mean_gross_roi": float(sub["gross_roi"].mean()),
            "mean_net_roi": float(sub["net_roi"].mean()),
            "mean_fee_drag": float(sub["fee_drag"].mean()),
            "mean_slip_drag": float(sub["slip_drag"].mean()),
            "median_alpha_capture": float(valid_cap["alpha_capture_rate"].median()) if valid_cap.height else float("nan"),
            "win_rate": float((sub["net_roi"] > 0).mean()),
            "scored_fraction": float(scored.height / sub.height) if sub.height else 0.0,
        })
    return rows


def main(universe: str = "model_top"):
    print(f"[decompose] loading trades for universe={universe}")
    df = load_trades(universe)
    print(f"  {df.height} rows, strategies: {df['strategy'].unique().to_list()}")

    plot_alpha_capture(df, universe)
    plot_friction(df, universe)

    rows = summary_table(df)
    for r in rows:
        print(f"  {r['strategy']:25s}  net_roi={r['mean_net_roi']:+.4f}  "
              f"cap={r['median_alpha_capture']:.2%}  win={r['win_rate']:.2%}")

    out_path = BACKTEST / f"decomposition_{universe}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"[decompose] saved {out_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "model_top")
