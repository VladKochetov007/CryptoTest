"""Trailing-stop parameter sweep on the model_top universe.

Answers: what trailing drawdown threshold (20/30/40/50/60%) and what arming
multiple (1.3x/1.5x/2.0x) maximize total PnL on the meta-OOF top-decile?

Grid: trail_frac in {0.8,0.7,0.6,0.5,0.4} x arm_mult in {1.3,1.5,2.0} = 15 cells.
sl_frac fixed at 0.4.

Outputs:
  eda/backtest/trailing_sweep.json     — per-cell stats
  eda/plots/trailing_grid.png          — heatmap total PnL
  eda/plots/trailing_winrate_grid.png  — heatmap win rate
  eda/plots/trailing_position_sweep.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from eda.backtest import (
        _run_trailing, simulate, load_token_panel, load_deployer_first_sell,
        _load_scores_and_actuals, TradeResult,
    )
except ModuleNotFoundError:
    from backtest import (
        _run_trailing, simulate, load_token_panel, load_deployer_first_sell,
        _load_scores_and_actuals, TradeResult,
    )

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts"
OUT = ROOT / "eda" / "backtest"
PLOTS = ROOT / "eda" / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

TRAIL_FRACS = [0.8, 0.7, 0.6, 0.5, 0.4]
ARM_MULTS = [1.3, 1.5, 2.0]
HARD_SL_FRAC = 0.4
POSITION_SIZES = [0.05, 0.10, 0.20, 0.50]


def cell_metrics(results: list[TradeResult]) -> dict:
    rois = np.clip(np.array([r.net_roi for r in results]), -1.0, 5.0)
    return {
        "n": len(results),
        "total_pnl_sol": float(rois.sum()),
        "mean_roi": float(rois.mean()),
        "median_roi": float(np.median(rois)),
        "win_rate": float((rois > 0).mean()),
    }


def select_universe() -> tuple[list[int], dict, dict, dict]:
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
    buy_ids = sel.tolist()
    panels = load_token_panel(buy_ids)
    sells = load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sells.get(tid)} for tid in panels}
    scores, actuals = _load_scores_and_actuals(buy_ids)
    return buy_ids, panels, extras, scores, actuals


def plot_heatmap(mat: np.ndarray, row_labels: list, col_labels: list,
                 row_axis: str, col_axis: str, title: str, path: Path, fmt: str = "+.0f"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_axis)
    ax.set_ylabel(row_axis)
    ax.set_title(title)
    vmax = np.abs(mat).max()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if abs(mat[i, j]) > 0.6 * vmax else "black"
            ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                    color=color, fontsize=10)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_position_sweep(best: dict, panels, extras, scores, actuals):
    arm_mult = best["arm_mult"]
    trail_frac = best["trail_frac"]
    pnl_by_size = []
    for pos_sol in POSITION_SIZES:
        def strategy(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras,
                     _am=arm_mult, _tf=trail_frac):
            return _run_trailing(tid, secs, prices, _am, _tf, HARD_SL_FRAC)
        res = simulate(strategy, panels, extras, scores, actuals)
        total_pnl = sum(r.net_roi * pos_sol for r in res)
        pnl_by_size.append(total_pnl)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(POSITION_SIZES)), pnl_by_size, color="steelblue")
    ax.set_xticks(range(len(POSITION_SIZES)))
    ax.set_xticklabels([f"{p:.2f} SOL" for p in POSITION_SIZES])
    ax.set_ylabel("total PnL (SOL)")
    ax.set_title(f"Position size sensitivity — arm={arm_mult:.1f}x trail={int((1-trail_frac)*100)}%")
    ax.axhline(0, color="red", linewidth=0.8)
    for i, v in enumerate(pnl_by_size):
        ax.text(i, v + 0.01 * (abs(v) + 0.1), f"{v:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = PLOTS / "trailing_position_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def main():
    print("[trailing_sweep] loading meta top-decile universe")
    buy_ids, panels, extras, scores, actuals = select_universe()
    print(f"  panels={len(panels)} scored={len(scores)} actuals={len(actuals)}")

    grid = []
    pnl_mat = np.zeros((len(ARM_MULTS), len(TRAIL_FRACS)))
    win_mat = np.zeros((len(ARM_MULTS), len(TRAIL_FRACS)))

    for i, arm in enumerate(ARM_MULTS):
        for j, frac in enumerate(TRAIL_FRACS):
            def strategy(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras,
                         _am=arm, _tf=frac):
                return _run_trailing(tid, secs, prices, _am, _tf, HARD_SL_FRAC)
            res = simulate(strategy, panels, extras, scores, actuals)
            m = cell_metrics(res)
            m["arm_mult"] = arm
            m["trail_frac"] = frac
            m["trail_drawdown_pct"] = int(round((1 - frac) * 100))
            grid.append(m)
            pnl_mat[i, j] = m["total_pnl_sol"]
            win_mat[i, j] = m["win_rate"]
            print(f"  arm={arm:.1f}x trail={int((1-frac)*100):>2}%  "
                  f"PnL={m['total_pnl_sol']:+8.1f}  win={m['win_rate']*100:5.1f}%  "
                  f"med_roi={m['median_roi']*100:+6.1f}%")

    (OUT / "trailing_sweep.json").write_text(json.dumps(grid, indent=2))

    arm_labels = [f"{a:.1f}x" for a in ARM_MULTS]
    trail_labels = [f"{int((1-f)*100)}%" for f in TRAIL_FRACS]
    plot_heatmap(pnl_mat, arm_labels, trail_labels,
                 "arm multiple", "trailing drawdown threshold",
                 "Total PnL (SOL, winsorized at +500%) — meta top-decile",
                 PLOTS / "trailing_grid.png")
    plot_heatmap(win_mat, arm_labels, trail_labels,
                 "arm multiple", "trailing drawdown threshold",
                 "Win rate — meta top-decile",
                 PLOTS / "trailing_winrate_grid.png", fmt=".2%")

    best = max(grid, key=lambda c: c["total_pnl_sol"])
    print(f"\n[best] arm={best['arm_mult']:.1f}x trail={best['trail_drawdown_pct']}%  "
          f"PnL={best['total_pnl_sol']:+.1f} SOL  win={best['win_rate']*100:.1f}%")

    plot_position_sweep(best, panels, extras, scores, actuals)


if __name__ == "__main__":
    main()
