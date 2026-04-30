"""Risk metrics and fixed-bankroll allocation checks for the Pump.fun backtest.

This script answers the practical capital question:

- how long the model-top test window is,
- how much PnL is made with 0.1 SOL fixed positions,
- what happens when a 1 SOL bankroll enforces max-open-position caps,
- and what annualized risk metrics look like on daily account returns.

It intentionally recomputes trades through `eda.backtest` instead of trusting stale
JSON artifacts. Output is written to `eda/backtest/risk_metrics.json`.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

try:
    from eda import backtest as bt
except ModuleNotFoundError:
    import backtest as bt


ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
OUT = ROOT / "eda" / "backtest"
OUT.mkdir(exist_ok=True)

STARTING_CAPITAL_SOL = 1.0
POSITION_SOL = bt.POSITION_SOL
SECONDS_PER_DAY = 86_400
TRADING_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    arm_mult: float
    trail_frac: float
    sl_frac: float = 0.4


STRATEGIES = [
    StrategySpec("trailing_30_default", arm_mult=1.5, trail_frac=0.7),
    StrategySpec("trailing_20_tight", arm_mult=1.3, trail_frac=0.8),
]


def select_model_top_universe(limit: int = 5000) -> list[int]:
    tid_path = bt.ART / "meta__token_ids.npy"
    oof_path = bt.ART / "meta__lgbm" / "oof_pred.npy"
    if not (tid_path.exists() and oof_path.exists()):
        raise RuntimeError("meta OOF artifacts missing; run eda/meta_train.py first")

    token_ids = np.load(tid_path)
    oof = np.load(oof_path)
    mask = ~np.isnan(oof)
    token_ids, oof = token_ids[mask], oof[mask]
    threshold = np.quantile(oof, 0.9)
    selected = token_ids[oof >= threshold]
    if len(selected) > limit:
        selected = np.random.default_rng(42).choice(selected, limit, replace=False)
    return [int(t) for t in selected]


def simulate_strategy(spec: StrategySpec, panels: dict[int, pl.DataFrame],
                      extras: dict, scores: dict[int, float],
                      actuals: dict[int, int]) -> pl.DataFrame:
    def strategy(tid, secs, prices, vol, buy_vol, sell_vol, top_w, holders, extras):
        return bt._run_trailing(tid, secs, prices, spec.arm_mult, spec.trail_frac, spec.sl_frac)

    results = bt.simulate(strategy, panels, extras, scores, actuals)
    if not results:
        return pl.DataFrame()

    trades = pl.DataFrame([asdict(r) for r in results]).with_columns(
        pl.lit(spec.name).alias("strategy"),
        (pl.col("net_roi") * POSITION_SOL).alias("pnl_sol"),
    )
    times = pl.read_parquet(FEAT).select("token_id", "deploy_time_unix")
    return trades.join(times, on="token_id", how="left").with_columns(
        (pl.col("deploy_time_unix") + pl.col("entry_sec")).alias("entry_abs"),
        (pl.col("deploy_time_unix") + pl.col("exit_sec")).alias("exit_abs"),
        (pl.col("entry_price") / pl.col("slot0_price") - 1.0).alias("entry_drift"),
        (pl.col("peak_price") / pl.col("entry_price")).alias("peak_mult"),
    )


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peaks, 1e-12) - 1.0
    return float(dd.min())


def profit_factor(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def daily_returns_from_events(events: list[tuple[int, float]], start_ts: int,
                              end_ts: int, starting_capital: float) -> np.ndarray:
    if end_ts <= start_ts:
        return np.array([], dtype=float)

    event_times = np.array([t for t, _ in events], dtype=np.int64)
    event_equity = np.array([e for _, e in events], dtype=float)
    day_start = int(start_ts // SECONDS_PER_DAY)
    day_end = int(end_ts // SECONDS_PER_DAY)
    closes = []
    for day in range(day_start, day_end + 1):
        ts = (day + 1) * SECONDS_PER_DAY - 1
        idx = np.searchsorted(event_times, ts, side="right") - 1
        closes.append(event_equity[idx] if idx >= 0 else starting_capital)
    closes = np.array(closes, dtype=float)
    if len(closes) < 2:
        return np.array([], dtype=float)
    return closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0


def annualized_stats(daily_returns: np.ndarray, final_equity: float,
                     starting_capital: float, period_days: float,
                     mdd: float) -> dict:
    if len(daily_returns) == 0:
        return {
            "daily_mean_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe_ann": float("nan"),
            "sortino_ann": float("nan"),
            "cagr": float("nan"),
            "calmar": float("nan"),
        }

    mean = float(daily_returns.mean())
    std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else float("nan")
    downside = daily_returns[daily_returns < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    cagr = (final_equity / starting_capital) ** (TRADING_DAYS_PER_YEAR / period_days) - 1.0

    return {
        "daily_mean_return": mean,
        "annualized_vol": std * np.sqrt(TRADING_DAYS_PER_YEAR) if std == std else float("nan"),
        "sharpe_ann": (mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std and std == std else float("nan"),
        "sortino_ann": (mean / down_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if down_std and down_std == down_std else float("nan"),
        "cagr": float(cagr),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("inf"),
    }


def apply_allocation_cap(trades: pl.DataFrame, max_open: int,
                         starting_capital: float = STARTING_CAPITAL_SOL,
                         position_sol: float = POSITION_SOL) -> dict:
    rows = trades.sort("entry_abs").select(
        "token_id", "entry_abs", "exit_abs", "pnl_sol", "net_roi"
    ).iter_rows(named=True)

    cash = starting_capital
    accepted = 0
    skipped = 0
    realized_pnl = 0.0
    open_heap: list[tuple[int, float, float]] = []
    events: list[tuple[int, float]] = []
    accepted_pnls: list[float] = []
    accepted_rois: list[float] = []
    max_concurrent = 0
    first_ts = None
    last_ts = None

    def equity_now() -> float:
        return cash + len(open_heap) * position_sol

    def close_until(ts: int) -> None:
        nonlocal cash, realized_pnl, last_ts
        while open_heap and open_heap[0][0] <= ts:
            exit_ts, returned_cash, pnl = heapq.heappop(open_heap)
            cash += returned_cash
            realized_pnl += pnl
            last_ts = exit_ts if last_ts is None else max(last_ts, exit_ts)
            events.append((exit_ts, equity_now()))

    for row in rows:
        entry_ts = int(row["entry_abs"])
        exit_ts = int(row["exit_abs"])
        pnl = float(row["pnl_sol"])
        net_roi = float(row["net_roi"])
        first_ts = entry_ts if first_ts is None else min(first_ts, entry_ts)
        close_until(entry_ts)

        if len(open_heap) >= max_open or cash + 1e-12 < position_sol:
            skipped += 1
            continue

        cash -= position_sol
        heapq.heappush(open_heap, (exit_ts, position_sol + pnl, pnl))
        accepted += 1
        accepted_pnls.append(pnl)
        accepted_rois.append(net_roi)
        max_concurrent = max(max_concurrent, len(open_heap))
        events.append((entry_ts, equity_now()))

    close_until(2**62)
    if first_ts is None:
        first_ts = 0
    if last_ts is None:
        last_ts = first_ts
    if not events:
        events = [(first_ts, starting_capital)]

    events.sort(key=lambda x: x[0])
    equity = np.array([e for _, e in events], dtype=float)
    pnls = np.array(accepted_pnls, dtype=float)
    rois = np.array(accepted_rois, dtype=float)
    period_days = max((last_ts - first_ts) / SECONDS_PER_DAY, 1e-9)
    daily_returns = daily_returns_from_events(events, first_ts, last_ts, starting_capital)
    mdd = max_drawdown(equity)
    final_equity = float(equity[-1])
    stats = annualized_stats(daily_returns, final_equity, starting_capital, period_days, mdd)

    return {
        "max_open": max_open,
        "starting_capital_sol": starting_capital,
        "position_sol": position_sol,
        "accepted_trades": accepted,
        "skipped_trades": skipped,
        "max_concurrent": max_concurrent,
        "period_days": period_days,
        "final_equity_sol": final_equity,
        "net_pnl_sol": float(final_equity - starting_capital),
        "return_on_initial_capital": float(final_equity / starting_capital - 1.0),
        "return_on_deployed_capital": float(pnls.sum() / (accepted * position_sol)) if accepted else float("nan"),
        "turnover_x_initial_capital": float(accepted * position_sol / starting_capital),
        "mean_trade_net_roi": float(rois.mean()) if len(rois) else float("nan"),
        "median_trade_net_roi": float(np.median(rois)) if len(rois) else float("nan"),
        "trade_win_rate": float((rois > 0).mean()) if len(rois) else float("nan"),
        "profit_factor": profit_factor(pnls),
        "max_drawdown": mdd,
        **stats,
    }


def unconstrained_summary(trades: pl.DataFrame) -> dict:
    entry = trades["entry_abs"].to_numpy()
    exit_ = trades["exit_abs"].to_numpy()
    pnl = trades["pnl_sol"].to_numpy()
    roi = trades["net_roi"].to_numpy()
    winsorized_roi = np.clip(roi, -1.0, 5.0)
    events = []
    for s, e in zip(entry, exit_):
        events.append((int(s), 1))
        events.append((int(e), -1))
    events.sort(key=lambda x: (x[0], x[1]))
    open_n = 0
    max_open = 0
    for _, delta in events:
        open_n += delta
        max_open = max(max_open, open_n)

    return {
        "n_trades": int(trades.height),
        "period_days": float((exit_.max() - entry.min()) / SECONDS_PER_DAY),
        "total_pnl_sol_at_0p1_sol": float(pnl.sum()),
        "winsorized_pnl_sol_cap500pct": float(winsorized_roi.sum() * POSITION_SOL),
        "capital_deployed_sol": float(trades.height * POSITION_SOL),
        "return_on_deployed_capital": float(pnl.sum() / (trades.height * POSITION_SOL)),
        "winsorized_return_on_deployed_capital": float(winsorized_roi.mean()),
        "mean_trade_net_roi": float(roi.mean()),
        "median_trade_net_roi": float(np.median(roi)),
        "trade_win_rate": float((roi > 0).mean()),
        "profit_factor": profit_factor(pnl),
        "max_concurrent_observed": int(max_open),
        "median_entry_drift": float(trades["entry_drift"].median()),
        "p90_entry_drift": float(trades["entry_drift"].quantile(0.9)),
        "median_hold_sec": float(trades["holding_sec"].median()),
        "p90_hold_sec": float(trades["holding_sec"].quantile(0.9)),
    }


def run() -> dict:
    buy_ids = select_model_top_universe()
    panels = bt.load_token_panel(buy_ids)
    sell_secs = bt.load_deployer_first_sell(buy_ids)
    extras = {tid: {"deployer_first_sell_sec": sell_secs.get(tid)} for tid in panels}
    scores, actuals = bt._load_scores_and_actuals(buy_ids)

    output = {
        "assumptions": {
            "position_sol": POSITION_SOL,
            "starting_capital_sol_for_cap_tests": STARTING_CAPITAL_SOL,
            "latency_sec": bt.LATENCY_SEC,
            "model_universe": "meta__lgbm OOF top decile, capped at 5000 tokens with seed 42",
            "annualization": "daily account returns, 365-day year; unstable because sample is ~16 days",
            "no_compounding": "position size is fixed at 0.1 SOL; profits are not used to increase size",
        },
        "strategies": {},
    }

    for spec in STRATEGIES:
        trades = simulate_strategy(spec, panels, extras, scores, actuals)
        trades.write_parquet(OUT / f"risk_trades_{spec.name}.parquet")
        cap_rows = [
            apply_allocation_cap(trades, max_open=n)
            for n in (1, 2, 3, 5, 7, 10, 20)
        ]
        output["strategies"][spec.name] = {
            "params": asdict(spec),
            "unconstrained": unconstrained_summary(trades),
            "allocation_caps": cap_rows,
        }

    out_path = OUT / "risk_metrics.json"
    safe_output = json_safe(output)
    out_path.write_text(json.dumps(safe_output, indent=2, default=str, allow_nan=False))
    return safe_output


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    data = run()
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
