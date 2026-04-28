"""Trailing-stop parameter sweep on the model_top universe.

Answers: what trailing drawdown threshold (20/30/40/50/60%) and what arming
multiple (1.3x/1.5x/2.0x) maximize total PnL on the meta-OOF top-decile?

Output:
  eda/backtest/trailing_sweep.json   — per-cell PnL / win-rate / median ROI
  eda/plots/trailing_grid.png        — heatmap of total PnL vs (arm_mult, trail_frac)

The hard SL is held fixed at 0.4*entry across the grid; the dimension being
swept is the trailing-from-peak threshold + when to arm it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from backtest import TradeResult, load_deployer_first_sell, load_token_panel, simulate

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts"
OUT = ROOT / "eda" / "backtest"
PLOTS = ROOT / "eda" / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

TRAIL_FRACS = [0.8, 0.7, 0.6, 0.5, 0.4]  # 20% / 30% / 40% / 50% / 60% drawdown
ARM_MULTS = [1.3, 1.5, 2.0]
HARD_SL_FRAC = 0.4


def make_trailing(arm_mult: float, trail_frac: float, hard_sl_frac: float = HARD_SL_FRAC):
    def fn(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
        entry = prices[0]
        cum_max = entry
        cum_min = entry
        armed_after = arm_mult * entry
        armed = False
        for i in range(1, len(prices)):
            cum_max = max(cum_max, prices[i])
            cum_min = min(cum_min, prices[i])
            if not armed and prices[i] >= armed_after:
                armed = True
            if armed and prices[i] <= trail_frac * cum_max:
                return TradeResult(tid, entry, prices[i], int(secs[i]),
                                    cum_min / entry - 1, cum_max / entry - 1,
                                    int(secs[i] - secs[0]), "trail")
            if prices[i] <= hard_sl_frac * entry:
                return TradeResult(tid, entry, prices[i], int(secs[i]),
                                    cum_min / entry - 1, cum_max / entry - 1,
                                    int(secs[i] - secs[0]), "sl_hard")
        return TradeResult(tid, entry, prices[-1], int(secs[-1]),
                            cum_min / entry - 1, cum_max / entry - 1,
                            int(secs[-1] - secs[0]), "timeout")
    return fn


def cell_metrics(results: list[TradeResult]) -> dict:
    rois = np.clip(np.array([r.roi for r in results]), -1.0, 5.0)
    wins = (rois > 0).astype(int)
    return {
        "n": len(results),
        "total_pnl_sol": float(rois.sum()),  # 1 SOL per trade, ROI winsorized at 5x
        "mean_roi": float(rois.mean()),
        "median_roi": float(np.median(rois)),
        "win_rate": float(wins.mean()),
    }


def select_universe() -> list[int]:
    """Top-decile of meta__lgbm OOF, capped at 5000 with seed 42."""
    tid_path = ART / "meta__token_ids.npy"
    oof_path = ART / "meta__lgbm" / "oof_pred.npy"
    if not (tid_path.exists() and oof_path.exists()):
        raise RuntimeError("meta artefacts missing — run meta_train.py first")
    tids = np.load(tid_path)
    oof = np.load(oof_path)
    mask = ~np.isnan(oof)
    tids, oof = tids[mask], oof[mask]
    cut = np.quantile(oof, 0.9)
    sel = tids[oof >= cut]
    rng = np.random.default_rng(42)
    if len(sel) > 5000:
        sel = rng.choice(sel, 5000, replace=False)
    return sel.tolist()


def main():
    print("[trailing_sweep] selecting meta top-decile universe")
    buy_ids = select_universe()
    print(f"[trailing_sweep] {len(buy_ids)} tokens")

    print("[trailing_sweep] loading panels")
    panels = load_token_panel(buy_ids)
    sells = load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sells.get(tid)} for tid in panels}
    print(f"[trailing_sweep] {len(panels)} panels with slot data")

    grid = []
    pnl_mat = np.zeros((len(ARM_MULTS), len(TRAIL_FRACS)))
    for i, arm in enumerate(ARM_MULTS):
        for j, frac in enumerate(TRAIL_FRACS):
            fn = make_trailing(arm, frac)
            res = simulate(fn, panels, extras)
            m = cell_metrics(res)
            m["arm_mult"] = arm
            m["trail_frac"] = frac
            m["trail_drawdown_pct"] = int(round((1 - frac) * 100))
            grid.append(m)
            pnl_mat[i, j] = m["total_pnl_sol"]
            print(f"  arm={arm:.1f} trail={int((1-frac)*100):>2}%  "
                  f"PnL={m['total_pnl_sol']:+8.1f} win={m['win_rate']*100:5.1f}%  "
                  f"med_roi={m['median_roi']*100:+6.1f}%")

    (OUT / "trailing_sweep.json").write_text(json.dumps(grid, indent=2))

    # heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pnl_mat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(TRAIL_FRACS)))
    ax.set_xticklabels([f"{int((1-f)*100)}%" for f in TRAIL_FRACS])
    ax.set_yticks(range(len(ARM_MULTS)))
    ax.set_yticklabels([f"{a:.1f}x" for a in ARM_MULTS])
    ax.set_xlabel("Trailing drawdown threshold (% from peak)")
    ax.set_ylabel("Arm multiple (price must hit X*entry first)")
    ax.set_title("Total PnL on meta top-decile universe (1 SOL/trade, ROI cap +500%)")
    for i in range(len(ARM_MULTS)):
        for j in range(len(TRAIL_FRACS)):
            ax.text(j, i, f"{pnl_mat[i, j]:+.0f}",
                     ha="center", va="center",
                     color="black" if abs(pnl_mat[i, j]) < pnl_mat.max() * 0.6 else "white",
                     fontsize=10)
    plt.colorbar(im, ax=ax, label="Total PnL (SOL)")
    plt.tight_layout()
    p = PLOTS / "trailing_grid.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")

    best = max(grid, key=lambda c: c["total_pnl_sol"])
    print(f"\n[best] arm={best['arm_mult']:.1f}x trail={best['trail_drawdown_pct']}%  "
          f"PnL={best['total_pnl_sol']:+.1f} SOL  win={best['win_rate']*100:.1f}%  "
          f"med_roi={best['median_roi']*100:+.1f}%")


if __name__ == "__main__":
    main()
