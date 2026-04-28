"""Model-as-exit: rerun the with60s model on a held position to decide stay/cut.

Policy on the meta top-decile universe:
  - Buy 1 SOL at slot 0.
  - At t = 60 s, score with the with60s/lgbm OOF probability for that token.
  - If with60s_p < EXIT_PROB_FLOOR -> exit immediately at t=60s with 100bps slip.
  - Otherwise, apply the trailing winner from trailing_sweep.py (arm=1.3, trail=20%, sl=40%).

This formalises the user's "rerun the model to track positions" idea.
Compares against the trailing-only baseline on the same universe and panels.

Output: eda/backtest/model_exit_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from backtest import TradeResult, load_deployer_first_sell, load_token_panel, simulate
from trailing_sweep import make_trailing, select_universe

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts"
OUT = ROOT / "eda" / "backtest"

EXIT_PROB_FLOOR = 0.15
SLIP_BPS = 100
ARM_MULT = 1.3
TRAIL_FRAC = 0.8
HARD_SL_FRAC = 0.4


def load_with60s_oof() -> dict[int, float]:
    tids = np.load(ART / "with60s__token_ids.npy")
    p = np.load(ART / "with60s__lgbm" / "oof_pred.npy")
    mask = ~np.isnan(p)
    return dict(zip(tids[mask].tolist(), p[mask].tolist()))


def make_model_exit(p60: dict[int, float]):
    """Returns a strategy_fn compatible with backtest.simulate()."""
    base = make_trailing(ARM_MULT, TRAIL_FRAC, HARD_SL_FRAC)
    slip = SLIP_BPS / 10_000

    def fn(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
        entry = prices[0]
        idx_60 = int(np.searchsorted(secs, 60, side="left"))
        tid_int = int(tid[0]) if isinstance(tid, tuple) else int(tid)
        prob = p60.get(tid_int)
        if prob is not None and idx_60 < len(prices) and prob < EXIT_PROB_FLOOR:
            exit_p_raw = prices[idx_60]
            exit_p = exit_p_raw * (1 - slip)
            return TradeResult(
                tid, entry, exit_p, int(secs[idx_60]),
                exit_p_raw / entry - 1, exit_p_raw / entry - 1,
                int(secs[idx_60] - secs[0]), "model_exit_60s",
            )
        return base(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras)

    return fn


def summarize(results: list[TradeResult]) -> dict:
    rois = np.clip(np.array([r.roi for r in results]), -1.0, 5.0)
    reasons = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return {
        "n": len(results),
        "total_pnl_sol": float(rois.sum()),
        "mean_roi": float(rois.mean()),
        "median_roi": float(np.median(rois)),
        "p10_roi": float(np.quantile(rois, 0.1)),
        "p90_roi": float(np.quantile(rois, 0.9)),
        "win_rate": float((rois > 0).mean()),
        "exit_reasons": reasons,
    }


def main():
    print("[model_exit] selecting meta top-decile universe")
    buy_ids = select_universe()
    print(f"[model_exit] {len(buy_ids)} tokens")

    p60 = load_with60s_oof()
    print(f"[model_exit] {len(p60)} tokens have with60s OOF")

    print("[model_exit] loading panels")
    panels = load_token_panel(buy_ids)
    sells = load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sells.get(tid)} for tid in panels}

    baseline_fn = make_trailing(ARM_MULT, TRAIL_FRAC, HARD_SL_FRAC)
    model_fn = make_model_exit(p60)

    print("[model_exit] running baseline trailing")
    base_res = simulate(baseline_fn, panels, extras)
    base_summary = summarize(base_res)

    print("[model_exit] running model-as-exit policy")
    me_res = simulate(model_fn, panels, extras)
    me_summary = summarize(me_res)

    out = {
        "params": {
            "arm_mult": ARM_MULT,
            "trail_frac": TRAIL_FRAC,
            "hard_sl_frac": HARD_SL_FRAC,
            "exit_prob_floor": EXIT_PROB_FLOOR,
            "slip_bps": SLIP_BPS,
            "stage1": "meta",
            "stage2": "with60s",
        },
        "trailing_only_baseline": base_summary,
        "model_as_exit": me_summary,
    }
    (OUT / "model_exit_summary.json").write_text(json.dumps(out, indent=2))

    diff = me_summary["total_pnl_sol"] - base_summary["total_pnl_sol"]
    print(f"\n[baseline] PnL={base_summary['total_pnl_sol']:+.1f} SOL  "
          f"win={base_summary['win_rate']*100:.1f}%  "
          f"med_roi={base_summary['median_roi']*100:+.1f}%")
    print(f"[model_ex] PnL={me_summary['total_pnl_sol']:+.1f} SOL  "
          f"win={me_summary['win_rate']*100:.1f}%  "
          f"med_roi={me_summary['median_roi']*100:+.1f}%")
    print(f"[delta] {diff:+.1f} SOL  ({'better' if diff > 0 else 'worse'})")


if __name__ == "__main__":
    main()
