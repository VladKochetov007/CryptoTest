"""Block-bootstrap the model_top + trailing_30 OOS equity curve.

1-day blocks (entry-time bins) are sampled with replacement to build B alternate
universes; each is replayed in calendar time. Output is a 5–95 percentile cone
on cumulative PnL, drawn over the original equity curve.

Output: eda/plots/backtest_equity_bootstrap.png
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

ROI_CAP = 5.0
N_BOOT = 500
BLOCK_SEC = 86_400


def main():
    feat_time = pl.read_parquet(ROOT / "eda" / "features.parquet").select(
        "token_id", "deploy_time_unix"
    )
    df = pl.read_parquet(BT / "trades_model_top_trailing_30.parquet")
    if df["token_id"].dtype == pl.List(pl.Int64):
        df = df.with_columns(pl.col("token_id").list.first().alias("token_id"))
    df = df.join(feat_time, on="token_id", how="inner").sort("deploy_time_unix")

    times = df["deploy_time_unix"].to_numpy()
    rois = np.clip(df["roi"].to_numpy(), -1.0, ROI_CAP)
    blocks = (times // BLOCK_SEC).astype(np.int64)
    block_ids = np.unique(blocks)

    rng = np.random.default_rng(42)
    grid_t = np.arange(times.min(), times.max() + 1, 3600)
    pnl_curves = np.zeros((N_BOOT, len(grid_t)))

    for b in range(N_BOOT):
        sample = rng.choice(block_ids, size=len(block_ids), replace=True)
        idx_lists = [np.where(blocks == bid)[0] for bid in sample]
        idx = np.concatenate(idx_lists) if idx_lists else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        order = np.argsort(times[idx])
        t_b = times[idx][order]
        r_b = rois[idx][order]
        cum = np.cumsum(r_b)
        # interpolate onto common time grid (step / nearest)
        # use right-aligned step: at time T, equity = sum of all closed trades with t <= T
        slot = np.searchsorted(t_b, grid_t, side="right") - 1
        slot = np.clip(slot, 0, len(cum) - 1)
        eq = np.where(slot < 0, 0.0, cum[slot])
        pnl_curves[b] = eq

    p5 = np.percentile(pnl_curves, 5, axis=0)
    p95 = np.percentile(pnl_curves, 95, axis=0)
    p50 = np.percentile(pnl_curves, 50, axis=0)

    # original (non-bootstrapped) curve
    orig_cum = np.cumsum(rois)

    fig, ax = plt.subplots(figsize=(12, 6))
    dts_grid = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in grid_t]
    dts_orig = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in times]
    ax.fill_between(dts_grid, p5, p95, color="#27ae60", alpha=0.20,
                     label=f"5–95% block-bootstrap cone (n={N_BOOT}, 1-day blocks)")
    ax.plot(dts_grid, p50, color="#16a085", linewidth=1.4,
             linestyle=":", label="bootstrap median")
    ax.plot(dts_orig, orig_cum, color="#27ae60", linewidth=2.0,
             label=f"actual model_top + trailing_30 (final +{orig_cum[-1]:.0f} SOL)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Entry time (UTC, walk-forward OOF period)")
    ax.set_ylabel("Cumulative PnL (SOL, ROI winsorized at +500%)")
    ax.set_title("Block-bootstrap robustness — model top-decile + trailing_30")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.tight_layout()
    out = PLOTS / "backtest_equity_bootstrap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    final_p5 = float(p5[-1])
    final_p50 = float(p50[-1])
    final_p95 = float(p95[-1])
    final_orig = float(orig_cum[-1])
    print(f"[plot] {out}")
    print(f"  actual final PnL : +{final_orig:.0f} SOL")
    print(f"  bootstrap median : +{final_p50:.0f} SOL")
    print(f"  bootstrap p5–p95 : +{final_p5:.0f} … +{final_p95:.0f} SOL")
    print(f"  prob(final > 0)  : {(pnl_curves[:, -1] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
