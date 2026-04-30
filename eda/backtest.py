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
    from eda.amm import round_trip_pnl
except ModuleNotFoundError:
    # supports `python eda/backtest.py ...` where sys.path points at eda/
    from amm import round_trip_pnl

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
ART = ROOT / "eda" / "artifacts"
SLOTS = ROOT / "slot_features_60m.parquet"
ACTS = ROOT / "deployer_actions_60m.parquet"
OUT = ROOT / "eda" / "backtest"
OUT.mkdir(exist_ok=True)

LATENCY_SEC = 1
POSITION_SOL = 0.1


def roi_from_prices(entry_price: float, exit_price: float) -> float:
    """Round-trip ROI in SOL units for fixed POSITION_SOL size."""
    _, pnl, _ = round_trip_pnl(entry_price, exit_price, POSITION_SOL)
    return pnl / POSITION_SOL


@dataclass
class TradeResult:
    token_id: int
    entry_price: float
    exit_price: float
    exit_sec: int
    roi: float
    max_drawdown: float
    max_runup: float
    holding_sec: int
    reason: str


def load_token_panel(token_ids: list[int]) -> dict[int, pl.DataFrame]:
    """Load per-token slot_features as a dict of polars DataFrames sorted by seconds_since_deploy.

    Each df has columns: seconds_since_deploy, price_sol_per_token, volume_sol,
    sell_volume_sol, buy_volume_sol, top_wallet_bought, holders_count.
    """
    df = pl.scan_parquet(SLOTS).filter(
        pl.col("token_id").is_in(token_ids) & (pl.col("price_sol_per_token") > 0)
    ).select(
        "token_id", "seconds_since_deploy", "price_sol_per_token",
        "volume_sol", "buy_volume_sol", "sell_volume_sol",
        "top_wallet_bought", "holders_count",
    ).sort(["token_id", "seconds_since_deploy"]).collect()
    # polars group_by yields tuple keys; unwrap
    return {t[0]: g.sort("seconds_since_deploy") for t, g in df.group_by("token_id")}


def load_deployer_first_sell(token_ids: list[int]) -> dict[int, int]:
    df = pl.scan_parquet(ACTS).filter(
        pl.col("token_id").is_in(token_ids)
        & pl.col("deployer_action").str.to_lowercase().str.contains("sell")
    ).group_by("token_id").agg(pl.col("seconds_since_deploy").min().alias("sec")).collect()
    return {r["token_id"]: int(r["sec"]) for r in df.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# strategies — each returns TradeResult given a per-token panel + extras
# ---------------------------------------------------------------------------

def simulate(strategy_fn: Callable, panels: dict[int, pl.DataFrame],
             extras: dict, max_sec: int = 1800) -> list[TradeResult]:
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
        secs = secs_full[entry_idx:]  # keep absolute seconds_since_deploy; deployer_sell_exit compares to absolute sell_sec
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
            out.append(result)
    return out


def fixed_2x(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    target = 2.0 * entry
    cum_max = entry
    cum_min = entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if prices[i] >= target:
            _, pnl, _ = round_trip_pnl(entry, prices[i], POSITION_SOL)
            return TradeResult(tid, entry, prices[i], int(secs[i]), pnl / POSITION_SOL,
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "tp_2x")
    _, pnl, _ = round_trip_pnl(entry, prices[-1], POSITION_SOL)
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), pnl / POSITION_SOL,
                       cum_min / entry - 1, cum_max / entry - 1,
                       int(secs[-1] - secs[0]), "timeout")


def tp2x_sl50(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    target = 2.0 * entry
    stop = 0.5 * entry
    cum_max = entry
    cum_min = entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if prices[i] >= target:
            _, pnl, _ = round_trip_pnl(entry, prices[i], POSITION_SOL)
            return TradeResult(tid, entry, prices[i], int(secs[i]), pnl / POSITION_SOL,
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= stop:
            _, pnl, _ = round_trip_pnl(entry, prices[i], POSITION_SOL)
            return TradeResult(tid, entry, prices[i], int(secs[i]), pnl / POSITION_SOL,
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sl_50")
    _, pnl, _ = round_trip_pnl(entry, prices[-1], POSITION_SOL)
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), pnl / POSITION_SOL,
                       cum_min / entry - 1, cum_max / entry - 1,
                       int(secs[-1] - secs[0]), "timeout")


def trailing_30(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    armed_after = 1.5 * entry
    armed = False
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if not armed and prices[i] >= armed_after:
            armed = True
        if armed and prices[i] <= 0.7 * cum_max:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "trail")
        if prices[i] <= 0.4 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sl_60")
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), roi_from_prices(entry, prices[-1]),
                       cum_min / entry - 1, cum_max / entry - 1,
                       int(secs[-1] - secs[0]), "timeout")


def deployer_sell_exit(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    """Exit on first deployer sell; otherwise hold to 30min, but apply 50% SL."""
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    sell_sec = extras.get("deployer_first_sell_sec")
    target = 3.0 * entry
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        if sell_sec is not None and secs[i] >= sell_sec:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "deployer_sell")
        if prices[i] >= target:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "tp_3x")
        if prices[i] <= 0.5 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sl_50")
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), roi_from_prices(entry, prices[-1]),
                       cum_min / entry - 1, cum_max / entry - 1,
                       int(secs[-1] - secs[0]), "timeout")


def volume_stagnation(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    """Exit if volume below 10% of running 60-s max for 10 consecutive slots."""
    entry = prices[0]
    cum_max = entry
    cum_min = entry
    rolling_max_vol = 0.0
    stag = 0
    rolling_window = []
    for i in range(1, len(prices)):
        cum_max = max(cum_max, prices[i])
        cum_min = min(cum_min, prices[i])
        rolling_window.append(vol[i])
        rolling_window = [v for v in rolling_window
                          if secs[i] - secs.tolist()[max(0, len(rolling_window)-60)] <= 60]
        rolling_max_vol = max(rolling_max_vol, vol[i])
        if rolling_max_vol > 0 and vol[i] < 0.1 * rolling_max_vol:
            stag += 1
        else:
            stag = 0
        if stag >= 10:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "vol_stagnation")
        if prices[i] >= 2.0 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= 0.5 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sl_50")
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), roi_from_prices(entry, prices[-1]),
                       cum_min / entry - 1, cum_max / entry - 1,
                       int(secs[-1] - secs[0]), "timeout")


def sell_pressure(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
    """Exit when sell volume > 1.5x buy volume for 5 slots, with SL/TP guards."""
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
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sell_pressure")
        if prices[i] >= 2.0 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "tp_2x")
        if prices[i] <= 0.5 * entry:
            return TradeResult(tid, entry, prices[i], int(secs[i]), roi_from_prices(entry, prices[i]),
                               cum_min / entry - 1, cum_max / entry - 1,
                               int(secs[i] - secs[0]), "sl_50")
    return TradeResult(tid, entry, prices[-1], int(secs[-1]), roi_from_prices(entry, prices[-1]),
                       cum_min / entry - 1, cum_max / entry - 1,
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
    rois = np.array([r.roi for r in results])
    holds = np.array([r.holding_sec for r in results])
    drawdowns = np.array([r.max_drawdown for r in results])
    reasons = [r.reason for r in results]
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


def main():
    import sys
    feats = pl.read_parquet(FEAT)
    universe = sys.argv[1] if len(sys.argv) > 1 else "model_top"
    # Build buy universe
    art_dir = ART / "instant__lgbm"
    selection_method = universe
    rng = np.random.default_rng(42)
    if universe == "model_top":
        # Prefer meta__lgbm OOF (pre-buy AUC 0.80) over instant__lgbm (0.77).
        # Fall back to instant if meta artefacts not yet generated.
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
        # original task heuristic: deposit > 1 SOL AND CEX-funded
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
    print(f"[buys] selection={selection_method} n={len(buy_ids)}")

    print("[buys] loading per-token panels…")
    panels = load_token_panel(buy_ids)
    print(f"[buys] panels loaded: {len(panels)} tokens have slot data")
    print("[buys] loading deployer first-sell timing…")
    sell_secs = load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sell_secs.get(tid)} for tid in panels}

    summaries = {}
    per_token = {}
    for name, fn in STRATEGIES.items():
        print(f"[strategy] {name}")
        res = simulate(fn, panels, extras)
        summaries[name] = aggregate(res)
        per_token[name] = [vars(r) | {"roi": r.roi} for r in res]

    out_path = OUT / f"summaries_{selection_method}.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    for name, recs in per_token.items():
        pl.DataFrame(recs).write_parquet(OUT / f"trades_{selection_method}_{name}.parquet")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
