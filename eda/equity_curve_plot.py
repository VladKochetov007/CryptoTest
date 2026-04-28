"""Out-of-sample equity curves on calendar time — single clean image.

All trades are OOS by construction — model_top universe is selected via walk-forward
OOF predictions, and the two-stage sim consumes the same OOF stream.

Per-trade ROI is winsorized at +500% to prevent a single tail outlier from swamping
the comparison. The two-stage curve uses real SOL PnL (no winsorization) since its
0.1/1.0 SOL sizing already bounds tail impact.

Output: eda/plots/backtest_equity_curve_oos.png
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "eda" / "backtest"
PLOTS = ROOT / "eda" / "plots"
PLOTS.mkdir(exist_ok=True)

ROI_CAP = 5.0  # +500% per trade


def load_with_time(path: Path, feat_time: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df["token_id"].dtype == pl.List(pl.Int64):
        df = df.with_columns(pl.col("token_id").list.first().alias("token_id"))
    return df.join(feat_time, on="token_id", how="inner").sort("deploy_time_unix")


def equity_winsorized(df: pl.DataFrame, cap: float = ROI_CAP) -> tuple[np.ndarray, np.ndarray]:
    times = df["deploy_time_unix"].to_numpy()
    pnl = np.clip(df["roi"].to_numpy(), -1.0, cap)
    return times, np.cumsum(pnl)


def equity_real_sol(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    times = df["deploy_time_unix"].to_numpy()
    return times, np.cumsum(df["pnl_sol"].to_numpy())


def to_dt(unix_arr: np.ndarray) -> np.ndarray:
    return np.array([datetime.fromtimestamp(int(t), tz=timezone.utc) for t in unix_arr])


def main():
    feat_time = pl.read_parquet(ROOT / "eda" / "features.parquet").select("token_id", "deploy_time_unix")

    fig, ax = plt.subplots(figsize=(12, 6))

    curves = [
        (BT / "trades_random_trailing_30.parquet",        "Random universe + trailing_30",       "#95a5a6", 1.4, "-",  "wins"),
        (BT / "trades_cex_heuristic_trailing_30.parquet", "CEX heuristic + trailing_30",         "#f39c12", 1.4, "-",  "wins"),
        (BT / "trades_model_top_sell_pressure_5.parquet", "Model top-decile + sell_pressure_5",  "#2980b9", 1.6, "-",  "wins"),
        (BT / "trades_model_top_trailing_30.parquet",     "Model top-decile + trailing_30",      "#27ae60", 1.8, "-",  "wins"),
    ]

    finals = []
    for path, label, color, lw, ls, _ in curves:
        df = load_with_time(path, feat_time)
        t, eq = equity_winsorized(df)
        ax.plot(to_dt(t), eq, color=color, linewidth=lw, linestyle=ls,
                label=f"{label}  (final +{eq[-1]:.0f} SOL)")
        finals.append((label, eq[-1]))

    two = load_with_time(ROOT / "eda" / "two_stage" / "trades.parquet", feat_time)
    t2, eq2 = equity_real_sol(two)
    ax.plot(to_dt(t2), eq2, color="#2c3e50", linewidth=2.2, linestyle="--",
            label=f"Two-stage probe→scale + trailing  (final +{eq2[-1]:.0f} SOL, 100 bps slip)")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Entry time (UTC, walk-forward OOF period)")
    ax.set_ylabel("Cumulative PnL (SOL)")
    ax.set_title(
        "Out-of-sample equity curves — 1 SOL/trade, ROI winsorized at +500% per trade\n"
        "Two-stage uses real 0.1/1.0 SOL sizing with 100 bps round-trip slippage"
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    plt.tight_layout()
    out = PLOTS / "backtest_equity_curve_oos.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")
    for label, val in finals:
        print(f"  {label}: {val:+.1f} SOL")
    print(f"  Two-stage: {eq2[-1]:+.1f} SOL")


if __name__ == "__main__":
    main()
