"""Generate backtest visualization plots from existing trade-level parquets and JSON summaries.

Outputs to eda/plots/:
- backtest_winrate_by_universe.png  — bar chart: win rate × strategy × universe
- backtest_median_roi_by_universe.png  — bar chart: median ROI × strategy × universe
- backtest_roi_distribution.png  — boxplot of ROI per strategy on model_top universe
- backtest_two_stage_pnl_curve.png  — cumulative PnL curve for two-stage sim
- backtest_hold_vs_roi.png  — scatter of holding time vs ROI for the winning strategy
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
BT = ROOT / "eda" / "backtest"
PLOTS = ROOT / "eda" / "plots"
PLOTS.mkdir(exist_ok=True)

STRATS = [
    "tp_2x_only",
    "tp_2x_sl_50",
    "trailing_30",
    "deployer_sell_exit",
    "vol_stagnation_10",
    "sell_pressure_5",
]
UNIVERSES = ["random", "cex_heuristic", "model_top"]
COLORS = {"random": "#95a5a6", "cex_heuristic": "#f39c12", "model_top": "#27ae60"}
LABELS = {"random": "Random", "cex_heuristic": "CEX heuristic", "model_top": "Model top-decile"}


def load_summaries() -> dict[str, dict]:
    out = {}
    for u in UNIVERSES:
        with open(BT / f"summaries_{u}.json") as f:
            out[u] = json.load(f)
    return out


def plot_winrate(summaries: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(STRATS))
    w = 0.27
    for i, u in enumerate(UNIVERSES):
        vals = [summaries[u][s]["win_rate"] for s in STRATS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[u], color=COLORS[u])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATS, rotation=15, ha="right")
    ax.set_ylabel("Win rate (ROI > 0)")
    ax.set_title("Win rate by exit strategy and universe (5000 trades each)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend()
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    p = PLOTS / "backtest_winrate_by_universe.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_median_roi(summaries: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(STRATS))
    w = 0.27
    for i, u in enumerate(UNIVERSES):
        vals = [summaries[u][s]["median_roi"] * 100 for s in STRATS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[u], color=COLORS[u])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + (0.4 if v >= 0 else -1.0), f"{v:+.1f}%",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATS, rotation=15, ha="right")
    ax.set_ylabel("Median ROI (%)")
    ax.set_title("Median per-trade ROI by exit strategy and universe")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend()
    plt.tight_layout()
    p = PLOTS / "backtest_median_roi_by_universe.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_roi_distribution_model_top():
    fig, ax = plt.subplots(figsize=(11, 5))
    data = []
    for s in STRATS:
        df = pl.read_parquet(BT / f"trades_model_top_{s}.parquet")
        roi = df["roi"].to_numpy()
        roi = np.clip(roi, -1, 5)
        data.append(roi)
    bp = ax.boxplot(data, labels=STRATS, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#27ae60")
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("ROI (clipped to [-100%, +500%])")
    ax.set_title("Per-trade ROI distribution on model top-decile universe (5000 trades)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p = PLOTS / "backtest_roi_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_two_stage_pnl_curve():
    df = pl.read_parquet(ROOT / "eda" / "two_stage" / "trades.parquet")
    pnl = df["pnl_sol"].to_numpy()
    cum = np.cumsum(pnl)
    median_path = np.cumsum(np.full_like(pnl, np.median(pnl)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(cum, color="#27ae60", linewidth=1)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].plot(median_path, color="gray", linestyle="--", linewidth=1, label="median path")
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Cumulative PnL (SOL)")
    axes[0].set_title(f"Two-stage sim: cumulative PnL = +{cum[-1]:.0f} SOL (5000 trades, 100bps slip)")
    axes[0].legend()

    sorted_pnl = np.sort(pnl)
    cdf_x = sorted_pnl
    cdf_y = np.arange(1, len(pnl) + 1) / len(pnl)
    axes[1].semilogx(np.abs(cdf_x[cdf_x < 0]), cdf_y[cdf_x < 0][::-1], color="#c0392b", label="losses")
    axes[1].semilogx(cdf_x[cdf_x > 0], 1 - cdf_y[cdf_x > 0], color="#27ae60", label="gains")
    axes[1].set_xlabel("|PnL| (SOL, log scale)")
    axes[1].set_ylabel("Fraction of trades")
    axes[1].set_title("PnL tail distribution — fat-right (tail-harvest profile)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOTS / "backtest_two_stage_pnl_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_hold_vs_roi():
    df = pl.read_parquet(BT / "trades_model_top_trailing_30.parquet")
    hold = df["holding_sec"].to_numpy()
    roi = df["roi"].to_numpy()
    reason = df["reason"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    color_map = {"trail": "#27ae60", "sl_60": "#c0392b", "timeout": "#7f8c8d"}
    for r, color in color_map.items():
        mask = reason == r
        ax.scatter(hold[mask], np.clip(roi[mask], -1, 5), s=4, alpha=0.4, c=color, label=f"{r} ({mask.sum()})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Holding time (s)")
    ax.set_ylabel("ROI (clipped at +500%)")
    ax.set_xscale("log")
    ax.set_title("trailing_30 on model top-decile: hold time vs ROI by exit reason")
    ax.legend()
    plt.tight_layout()
    p = PLOTS / "backtest_hold_vs_roi.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def main():
    summaries = load_summaries()
    plot_winrate(summaries)
    plot_median_roi(summaries)
    plot_roi_distribution_model_top()
    plot_two_stage_pnl_curve()
    plot_hold_vs_roi()
    print("[done]")


if __name__ == "__main__":
    main()
