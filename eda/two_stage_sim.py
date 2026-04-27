"""End-to-end two-stage simulator.

Stage 1 (slot 0): instant model score ≥ buy_threshold → probe size.
Stage 2 (60 s): with60s model score evaluated. If ≥ scale_threshold → scale to full.
                If < abort_threshold → exit at slot-60 price.
                Otherwise keep probe to exit policy.
Stage 3 (exit policy applied to running position): trailing-30 (best from Part 2).

Reports per-token PnL and aggregated portfolio PnL with cost model.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
ART = ROOT / "eda" / "artifacts"
SLOTS = ROOT / "slot_features_60m.parquet"
OUT = ROOT / "eda" / "two_stage"
OUT.mkdir(exist_ok=True)


PROBE_SIZE = 0.1            # SOL
FULL_SIZE = 1.0             # SOL on scale
ENTRY_SLIPPAGE_BPS = 100    # 1%
EXIT_SLIPPAGE_BPS = 100     # 1%

BUY_THRESHOLD = 0.39
SCALE_THRESHOLD = 0.85
ABORT_THRESHOLD = 0.30

CAT_COLS = {"deployer_wallet_source_cex_name"}


def load_oof(name: str):
    tids = np.load(ART / f"{name}__token_ids.npy")
    y = np.load(ART / f"{name}__y.npy")
    p = np.load(ART / f"{name}__lgbm" / "oof_pred.npy")
    mask = ~np.isnan(p)
    return tids[mask], y[mask], p[mask]


def trailing_30_exit(prices: np.ndarray, secs: np.ndarray, entry_idx: int) -> tuple[int, float, str]:
    """From entry_idx onward, exit on trailing-30 / sl_60 / timeout."""
    entry = prices[entry_idx]
    cum_max = entry
    armed_after = 1.5 * entry
    armed = False
    for i in range(entry_idx + 1, len(prices)):
        cum_max = max(cum_max, prices[i])
        if not armed and prices[i] >= armed_after:
            armed = True
        if armed and prices[i] <= 0.7 * cum_max:
            return i, prices[i], "trail"
        if prices[i] <= 0.4 * entry:
            return i, prices[i], "sl_60"
    return len(prices) - 1, prices[-1], "timeout"


def slip(price: float, bps: int, side: str) -> float:
    f = bps / 10_000
    if side == "buy":
        return price * (1 + f)
    return price * (1 - f)


def simulate_one(prices: np.ndarray, secs: np.ndarray,
                  instant_p: float, with60s_p: float | None) -> dict:
    """Single token end-to-end."""
    if instant_p < BUY_THRESHOLD:
        return {"action": "skip", "pnl_sol": 0.0, "size_sol": 0.0, "exit_reason": "no_buy"}
    # stage 1: probe
    entry_p = slip(prices[0], ENTRY_SLIPPAGE_BPS, "buy")
    position_size = PROBE_SIZE
    avg_entry = entry_p

    # find slot at ~60 s for stage 2
    idx_60 = int(np.searchsorted(secs, 60))
    idx_60 = min(idx_60, len(prices) - 1)
    if with60s_p is not None and idx_60 < len(prices) - 1:
        if with60s_p < ABORT_THRESHOLD:
            exit_price = slip(prices[idx_60], EXIT_SLIPPAGE_BPS, "sell")
            pnl = (exit_price - avg_entry) * position_size / avg_entry
            return {"action": "abort_60s", "pnl_sol": pnl, "size_sol": position_size,
                    "exit_reason": "abort_60s",
                    "entry_price": avg_entry, "exit_price": exit_price}
        if with60s_p >= SCALE_THRESHOLD:
            scale_p = slip(prices[idx_60], ENTRY_SLIPPAGE_BPS, "buy")
            scale_add = FULL_SIZE - PROBE_SIZE
            avg_entry = (avg_entry * PROBE_SIZE + scale_p * scale_add) / FULL_SIZE
            position_size = FULL_SIZE

    # stage 3: trailing exit from idx_60 (or 0 if no scale)
    start = idx_60 if (with60s_p is not None and with60s_p >= SCALE_THRESHOLD) else 0
    exit_idx, raw_exit, reason = trailing_30_exit(prices, secs, start)
    exit_p = slip(raw_exit, EXIT_SLIPPAGE_BPS, "sell")
    pnl = (exit_p - avg_entry) * position_size / avg_entry
    return {"action": "full" if position_size == FULL_SIZE else "probe_only",
            "pnl_sol": pnl, "size_sol": position_size,
            "exit_reason": reason,
            "entry_price": avg_entry, "exit_price": exit_p,
            "hold_sec": int(secs[exit_idx] - secs[0])}


def main():
    tids_inst, y_inst, p_inst = load_oof("instant")
    tids_60, y_60, p_60 = load_oof("with60s")

    inst_map = dict(zip(tids_inst.tolist(), p_inst.tolist()))
    p60_map = dict(zip(tids_60.tolist(), p_60.tolist()))

    # only buys
    buy_ids = [t for t, p in inst_map.items() if p >= BUY_THRESHOLD]
    print(f"[2stage] {len(buy_ids)} buy candidates (instant p ≥ {BUY_THRESHOLD})")
    rng = np.random.default_rng(42)
    if len(buy_ids) > 5000:
        buy_ids = rng.choice(buy_ids, 5000, replace=False).tolist()
        print(f"[2stage] capped to 5000 for sim tractability")

    print("[2stage] loading panels…")
    panels_lf = pl.scan_parquet(SLOTS).filter(
        pl.col("token_id").is_in(buy_ids) & (pl.col("price_sol_per_token") > 0)
    ).select("token_id", "seconds_since_deploy", "price_sol_per_token").sort(
        ["token_id", "seconds_since_deploy"]
    ).collect()
    # group_by returns tuple keys → unwrap
    panels = {(int(k[0]) if isinstance(k, tuple) else int(k)): g
              for k, g in panels_lf.group_by("token_id")}
    print(f"[2stage] {len(panels)} panels loaded")

    results = []
    for tid in buy_ids:
        if tid not in panels:
            continue
        panel = panels[tid].sort("seconds_since_deploy")
        prices = panel["price_sol_per_token"].to_numpy()
        secs = panel["seconds_since_deploy"].to_numpy()
        out = simulate_one(prices, secs, inst_map[tid], p60_map.get(tid))
        out["token_id"] = int(tid)
        out["instant_p"] = float(inst_map[tid])
        out["with60s_p"] = float(p60_map.get(tid, np.nan))
        results.append(out)

    # normalise schema (some rows lack hold_sec/entry_price etc.)
    keys = ["token_id", "instant_p", "with60s_p", "action", "exit_reason",
            "pnl_sol", "size_sol", "entry_price", "exit_price", "hold_sec"]
    norm = [{k: r.get(k) for k in keys} for r in results]
    print(f"[2stage] results: {len(results)}, sample={results[:2]}")
    pls = pl.DataFrame(norm, infer_schema_length=len(norm))
    print(f"[2stage] cols={pls.columns}")
    pls.write_parquet(OUT / "trades.parquet")

    pnls = pls["pnl_sol"].to_numpy()
    summary = {
        "n_candidates": len(buy_ids),
        "n_simulated": len(results),
        "actions": dict(pls["action"].value_counts().iter_rows()),
        "exit_reasons": dict(pls["exit_reason"].value_counts().iter_rows()),
        "total_pnl_sol": float(pnls.sum()),
        "mean_pnl_sol": float(pnls.mean()),
        "median_pnl_sol": float(np.median(pnls)),
        "win_rate": float((pnls > 0).mean()),
        "p10_pnl_sol": float(np.quantile(pnls, 0.1)),
        "p90_pnl_sol": float(np.quantile(pnls, 0.9)),
        "worst_pnl_sol": float(pnls.min()),
        "best_pnl_sol": float(pnls.max()),
        "params": {
            "probe_size_sol": PROBE_SIZE, "full_size_sol": FULL_SIZE,
            "buy_threshold": BUY_THRESHOLD, "scale_threshold": SCALE_THRESHOLD,
            "abort_threshold": ABORT_THRESHOLD,
            "entry_slip_bps": ENTRY_SLIPPAGE_BPS, "exit_slip_bps": EXIT_SLIPPAGE_BPS,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
