"""Marimo dashboard for fixed-bankroll risk metrics."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    ROOT = Path("/home/vlad/development/PumpTest")
    RISK_JSON = ROOT / "eda" / "backtest" / "risk_metrics.json"
    return RISK_JSON, json, mo, pl


@app.cell
def _(RISK_JSON, json, mo):
    if not RISK_JSON.exists():
        mo.stop(True, mo.md("Run `python eda/risk_metrics.py` first."))
    risk = json.loads(RISK_JSON.read_text())
    mo.md(
        """
        # Fixed-bankroll risk check

        The backtest uses fixed `0.1 SOL` positions. This dashboard reads
        `eda/backtest/risk_metrics.json`, including the `1 SOL` bankroll scenario
        with `max_open` caps.
        """
    )
    return (risk,)


@app.cell
def _(mo, risk):
    assumptions = risk["assumptions"]
    mo.vstack([
        mo.md("## Assumptions"),
        assumptions,
    ])
    return


@app.cell
def _(pl, risk):
    rows = []
    for strategy, payload in risk["strategies"].items():
        row = {"strategy": strategy}
        row.update(payload["unconstrained"])
        rows.append(row)
    unconstrained = pl.DataFrame(rows)
    unconstrained
    return (unconstrained,)


@app.cell
def _(mo, unconstrained):
    mo.vstack([
        mo.md("## Unconstrained fixed-size runs"),
        unconstrained.select(
            "strategy",
            "n_trades",
            "period_days",
            "total_pnl_sol_at_0p1_sol",
            "winsorized_pnl_sol_cap500pct",
            "capital_deployed_sol",
            "return_on_deployed_capital",
            "winsorized_return_on_deployed_capital",
            "mean_trade_net_roi",
            "trade_win_rate",
            "profit_factor",
            "max_concurrent_observed",
        ),
    ])
    return


@app.cell
def _(pl, risk):
    cap_rows = []
    for strategy, payload in risk["strategies"].items():
        for row in payload["allocation_caps"]:
            cap_rows.append({"strategy": strategy, **row})
    caps = pl.DataFrame(cap_rows)
    caps
    return (caps,)


@app.cell
def _(caps, mo):
    mo.vstack([
        mo.md("## 1 SOL bankroll with max-open-position caps"),
        caps.select(
            "strategy",
            "max_open",
            "accepted_trades",
            "skipped_trades",
            "final_equity_sol",
            "net_pnl_sol",
            "return_on_initial_capital",
            "return_on_deployed_capital",
            "max_drawdown",
            "sharpe_ann",
            "sortino_ann",
            "profit_factor",
        ),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
