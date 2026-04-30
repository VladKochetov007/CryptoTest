"""Part 2 — exit-strategy simulator (honest execution).

Reads slot_features_60m and deployer_actions_60m, replays per-token tick streams,
and applies exit rules to a buy executed after fixed latency (skip devbuy slot0).
ROI computed via Pump.fun bonding-curve constant-product impact model + 1% fee/side.

Selection of "buys" mimics a real bot: top-decile predictions from the LGBM model
trained in `train.py` (instant feature set). Falls back to all tokens if model
artefacts are missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

try:
    from eda.amm import round_trip_pnl, TAKER_FEE
except ModuleNotFoundError:
    from amm import round_trip_pnl, TAKER_FEE

try:
    import numba as nb

    @nb.njit(cache=True)
    def _trailing_inner(prices: np.ndarray, arm_mult: float, trail_frac: float, sl_frac: float):
        """Tight trailing-stop loop. Returns (exit_idx, reason): 0=trail, 1=sl_hard, 2=timeout."""
        cum_max = prices[0]
        armed = False
        for i in range(1, len(prices)):
            if prices[i] > cum_max:
                cum_max = prices[i]
            if not armed and prices[i] >= arm_mult * prices[0]:
                armed = True
            if armed and prices[i] <= trail_frac * cum_max:
                return i, 0
            if prices[i] <= sl_frac * prices[0]:
                return i, 1
        return len(prices) - 1, 2

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False

    def _trailing_inner(prices, arm_mult, trail_frac, sl_frac):
        cum_max = prices[0]
        armed = False
        for i in range(1, len(prices)):
            if prices[i] > cum_max:
                cum_max = prices[i]
            if not armed and prices[i] >= arm_mult * prices[0]:
                armed = True
            if armed and prices[i] <= trail_frac * cum_max:
                return i, 0
            if prices[i] <= sl_frac * prices[0]:
                return i, 1
        return len(prices) - 1, 2


_TRAIL_REASONS = {0: "trail", 1: "sl_hard", 2: "timeout"}
_FLAT_FEE_DRAG = 2.0 * TAKER_FEE  # 2% round-trip fee (flat approximation)

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
ART = ROOT / "eda" / "artifacts"
SLOTS = ROOT / "slot_features_60m.parquet"
ACTS = ROOT / "deployer_actions_60m.parquet"
OUT = ROOT / "eda" / "backtest"
OUT.mkdir(exist_ok=True)

LATENCY_SEC = 1
POSITION_SOL = 0.1


@dataclass
class TradeResult:
    token_id: int
    decision_score: float      # model OOF probability (NaN if not scored)
    entry_sec: int             # seconds_since_deploy at entry
    entry_price: float
    slot0_price: float         # first available slot price (pre-entry)
    peak_price: float          # ATH during hold period
    exit_sec: int
    exit_price: float
    exit_reason: str
    gross_roi: float           # exit/entry - 1 (no friction)
    net_roi: float             # after AMM fees + price impact
    fee_drag: float            # flat 2 * TAKER_FEE per round trip
    slip_drag: float           # AMM price impact beyond flat fee
    max_drawdown: float        # cum_min/entry - 1
    holding_sec: int
    hit_2x_actual: int         # ground-truth label from features.parquet (-1 = unknown)


def _make_result(tid: int, entry: float, exit_p: float, exit_sec_abs: float,
                 cum_min: float, cum_max: float, hold_sec: int, reason: str) -> TradeResult:
    """Build TradeResult for a completed trade; simulate() fills decision_score etc."""
    _, pnl, _ = round_trip_pnl(entry, exit_p, POSITION_SOL)
    net_roi = pnl / POSITION_SOL
    gross_roi = exit_p / entry - 1.0
    return TradeResult(
        token_id=tid,
        decision_score=float("nan"),
        entry_sec=0,
        entry_price=entry,
        slot0_price=float("nan"),
        peak_price=cum_max,
        exit_sec=int(exit_sec_abs),
        exit_price=exit_p,
        exit_reason=reason,
        gross_roi=gross_roi,
        net_roi=net_roi,
        fee_drag=_FLAT_FEE_DRAG,
        slip_drag=gross_roi - net_roi - _FLAT_FEE_DRAG,
        max_drawdown=cum_min / entry - 1.0,
        holding_sec=hold_sec,
        hit_2x_actual=-1,
    )


def load_token_panel(token_ids: list[int]) -> dict[int, pl.DataFrame]:
    df = pl.scan_parquet(SLOTS).filter(
        pl.col("token_id").is_in(token_ids) & (pl.col("price_sol_per_token") > 0)
    ).select(
        "token_id", "seconds_since_deploy", "price_sol_per_token",
        "volume_sol", "buy_volume_sol", "sell_volume_sol",
        "top_wallet_bought", "holders_count",
    ).sort(["token_id", "seconds_since_deploy"]).collect()
    return {t[0]: g.sort("seconds_since_deploy") for t, g in df.group_by("token_id")}


def load_deployer_first_sell(token_ids: list[int]) -> dict[int, int]:
    df = pl.scan_parquet(ACTS).filter(
        pl.col("token_id").is_in(token_ids)
        & pl.col("deployer_action").str.to_lowercase().str.contains("sell")
    ).group_by("token_id").agg(pl.col("seconds_since_deploy").min().alias("sec")).collect()
    return {r["token_id"]: int(r["sec"]) for r in df.iter_rows(named=True)}


def simulate(
    strategy_fn: Callable,
    panels: dict[int, pl.DataFrame],
    extras: dict,
    scores: dict[int, float] | None = None,
    actuals: dict[int, int] | None = None,
    max_sec: int = 1800,
) -> list[TradeResult]:
    scores = scores or {}
    actuals = actuals or {}
    out = []
    for tid, panel in panels.items():
        if panel.height == 0:
            continue
        panel = panel.filter(pl.col("seconds_since_deploy") <= max_sec)
        if panel.height == 0:
            continue
        prices_full = panel["price_sol_per_token"].to_numpy()
        secs_full = panel["seconds_since_deploy"].to_numpy()
        entry_idx = int(np.searchsorted(secs_full, LATENCY_SEC, side="left"))
        if entry_idx >= len(prices_full):
            continue
        prices = prices_full[entry_idx:]
        secs = secs_full[entry_idx:]
        vol = panel["volume_sol"].to_numpy()[entry_idx:]
        buy_vol = panel["buy_volume_sol"].to_numpy()[entry_idx:]
        sell_vol = panel["sell_volume_sol"].to_numpy()[entry_idx:]
        top_w = panel["top_wallet_bought"].to_numpy().astype(bool)[entry_idx:]
        holders = panel["holders_count"].to_numpy()[entry_idx:]
        result = strategy_fn(
            tid=tid, secs=secs, prices=prices, vol=vol,
            buy_vol=buy_vol, sell_vol=sell_vol, top_w=top_w, holders=holders,
            extras=extras.get(tid, {}),
        )
        if result is not None:
            result.decision_score = scores.get(tid, float("nan"))
            result.entry_sec = int(secs[0])
            result.slot0_price = float(prices_full[0])
            result.hit_2x_actual = actuals.get(tid, -1)
            out.append(result)
    return out


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

def fixed_2x(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if prices[i] >= 2.0 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "tp_2x")
    return _make_result(tid, entry, prices[-1], secs[-1], cum_min, cum_max,
                        int(secs[-1] - secs[0]), "timeout")


def tp2x_sl50(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if prices[i] >= 2.0 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= 0.5 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "sl_50")
    return _make_result(tid, entry, prices[-1], secs[-1], cum_min, cum_max,
                        int(secs[-1] - secs[0]), "timeout")


def trailing_30(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    return _run_trailing(tid, secs, prices, arm_mult=1.5, trail_frac=0.7, sl_frac=0.4)


def _run_trailing(tid, secs, prices, arm_mult, trail_frac, sl_frac):
    arr = np.ascontiguousarray(prices, dtype=np.float64)
    exit_idx, code = _trailing_inner(arr, arm_mult, trail_frac, sl_frac)
    entry = prices[0]
    slice_ = prices[:exit_idx + 1]
    return _make_result(tid, entry, prices[exit_idx], secs[exit_idx],
                        slice_.min(), slice_.max(),
                        int(secs[exit_idx] - secs[0]), _TRAIL_REASONS[code])


def deployer_sell_exit(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    sell_sec = extras.get("deployer_first_sell_sec")
    target = 3.0 * entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if sell_sec is not None and secs[i] >= sell_sec:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "deployer_sell")
        if prices[i] >= target:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "tp_3x")
        if prices[i] <= 0.5 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "sl_50")
    return _make_result(tid, entry, prices[-1], secs[-1], cum_min, cum_max,
                        int(secs[-1] - secs[0]), "timeout")


def volume_stagnation(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    rolling_max_vol = 0.0
    stag = 0
    secs_list = secs.tolist()
    rolling_window: list[float] = []
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        rolling_window.append(vol[i])
        rolling_window = [v for v in rolling_window
                          if secs_list[i] - secs_list[max(0, len(rolling_window) - 60)] <= 60]
        rolling_max_vol = max(rolling_max_vol, vol[i])
        stag = stag + 1 if rolling_max_vol > 0 and vol[i] < 0.1 * rolling_max_vol else 0
        if stag >= 10:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "vol_stagnation")
        if prices[i] >= 2.0 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= 0.5 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "sl_50")
    return _make_result(tid, entry, prices[-1], secs[-1], cum_min, cum_max,
                        int(secs[-1] - secs[0]), "timeout")


def sell_pressure(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    consec = 0
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if buy_vol[i] > 0 and sell_vol[i] > 1.5 * buy_vol[i]:
            consec += 1
        elif buy_vol[i] == 0 and sell_vol[i] > 0:
            consec += 1
        else:
            consec = 0
        if consec >= 5:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "sell_pressure")
        if prices[i] >= 2.0 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= 0.5 * entry:
            return _make_result(tid, entry, prices[i], secs[i], cum_min, cum_max,
                                int(secs[i] - secs[0]), "sl_50")
    return _make_result(tid, entry, prices[-1], secs[-1], cum_min, cum_max,
                        int(secs[-1] - secs[0]), "timeout")


STRATEGIES = {
    "tp_2x_only": fixed_2x,
    "tp_2x_sl_50": tp2x_sl50,
    "trailing_30": trailing_30,
    "deployer_sell_exit": deployer_sell_exit,
    "vol_stagnation_10": volume_stagnation,
    "sell_pressure_5": sell_pressure,
}


def aggregate(results: list[TradeResult]) -> dict:
    if not results:
        return {}
    rois = np.array([r.net_roi for r in results])
    holds = np.array([r.holding_sec for r in results])
    drawdowns = np.array([r.max_drawdown for r in results])
    reasons = [r.exit_reason for r in results]
    reason_counts = {k: reasons.count(k) for k in set(reasons)}
    wins = rois > 0
    pos = np.sort(rois[rois > 0])[::-1]
    top10_share = float(pos[:min(10, len(pos))].sum() / pos.sum()) if len(pos) else float("nan")
    winsorized_pnl_sol = float(np.clip(rois, -1.0, 5.0).sum() * POSITION_SOL)
    return {
        "n": len(results),
        "mean_roi": float(rois.mean()),
        "median_roi": float(np.median(rois)),
        "p10_roi": float(np.quantile(rois, 0.1)),
        "p90_roi": float(np.quantile(rois, 0.9)),
        "win_rate": float(wins.mean()),
        "total_pnl_sol": float(rois.sum() * POSITION_SOL),
        "winsorized_pnl_sol_cap500pct": winsorized_pnl_sol,
        "top10_positive_pnl_share": top10_share,
        "median_hold_sec": float(np.median(holds)),
        "p90_hold_sec": float(np.quantile(holds, 0.9)),
        "median_dd": float(np.median(drawdowns)),
        "worst_dd": float(drawdowns.min()),
        "exit_reasons": reason_counts,
    }


def _load_scores_and_actuals(buy_ids: list[int]) -> tuple[dict[int, float], dict[int, int]]:
    scores: dict[int, float] = {}
    for fs in ("meta", "instant"):
        tid_path = ART / f"{fs}__token_ids.npy"
        oof_path = ART / f"{fs}__lgbm" / "oof_pred.npy"
        if tid_path.exists() and oof_path.exists():
            tids = np.load(tid_path)
            oof = np.load(oof_path)
            buy_set = set(buy_ids)
            for t, p in zip(tids, oof):
                if int(t) in buy_set and not np.isnan(p):
                    scores[int(t)] = float(p)
            break

    actuals: dict[int, int] = {}
    if FEAT.exists():
        feat_df = pl.scan_parquet(FEAT).filter(
            pl.col("token_id").is_in(buy_ids) & pl.col("hit_2x").is_not_null()
        ).select("token_id", "hit_2x").collect()
        for r in feat_df.iter_rows(named=True):
            actuals[int(r["token_id"])] = int(r["hit_2x"])

    return scores, actuals


def main():
    import sys
    feats = pl.read_parquet(FEAT)
    universe = sys.argv[1] if len(sys.argv) > 1 else "model_top"
    rng = np.random.default_rng(42)

    if universe == "model_top":
        for fs in ("meta", "instant"):
            tid_path = ART / f"{fs}__token_ids.npy"
            oof_path = ART / f"{fs}__lgbm" / "oof_pred.npy"
            if tid_path.exists() and oof_path.exists():
                print(f"[buys] universe model_top sourced from {fs}__lgbm OOF")
                token_ids = np.load(tid_path)
                oof = np.load(oof_path)
                break
        else:
            raise RuntimeError("model_top requested but no OOF artefacts found")
        mask = ~np.isnan(oof)
        token_ids, oof = token_ids[mask], oof[mask]
        thresh = np.quantile(oof, 0.9)
        sel = token_ids[oof >= thresh]
        if len(sel) > 5000:
            sel = rng.choice(sel, 5000, replace=False)
        buy_ids = sel.tolist()
    elif universe == "random":
        rows_with_label = feats.filter(pl.col("hit_2x").is_not_null())["token_id"].to_list()
        buy_ids = rng.choice(rows_with_label, min(5000, len(rows_with_label)), replace=False).tolist()
    elif universe == "cex_heuristic":
        rows = feats.filter(
            (pl.col("hit_2x").is_not_null())
            & (pl.col("deployer_deposit_amount") > 1.0)
            & (pl.col("is_cex") == 1)
        )["token_id"].to_list()
        if len(rows) > 5000:
            rows = rng.choice(rows, 5000, replace=False).tolist()
        buy_ids = rows
    else:
        raise ValueError(f"unknown universe {universe}")

    print(f"[buys] selection={universe} n={len(buy_ids)}")
    print("[buys] loading per-token panels...")
    panels = load_token_panel(buy_ids)
    print(f"[buys] panels loaded: {len(panels)} tokens have slot data")
    sell_secs = load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sell_secs.get(tid)} for tid in panels}
    scores, actuals = _load_scores_and_actuals(buy_ids)
    print(f"[buys] scored={len(scores)} actuals={len(actuals)}")

    summaries = {}
    trade_frames: list[pl.DataFrame] = []
    for name, fn in STRATEGIES.items():
        print(f"[strategy] {name}")
        res = simulate(fn, panels, extras, scores, actuals)
        summaries[name] = aggregate(res)
        df = pl.DataFrame([vars(r) for r in res]).with_columns(
            pl.lit(name).alias("strategy")
        )
        df.write_parquet(OUT / f"trades_{universe}_{name}.parquet")
        trade_frames.append(df)

    if trade_frames:
        combined = pl.concat(trade_frames)
        csv_path = OUT / f"trades_{universe}.csv"
        combined.write_csv(csv_path)
        print(f"[log] trade CSV: {csv_path} ({combined.height} rows)")

    out_path = OUT / f"summaries_{universe}.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
