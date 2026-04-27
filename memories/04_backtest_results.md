# Backtest Results

## Setup

- **Universe**: top-decile OOF LGBM instant predictions → 5,000 tokens (capped)
- **Baselines**: random 5,000 tokens; CEX heuristic (`is_cex=1` AND `deposit > 1 SOL`) 5,000 tokens
- **Entry**: first-slot price (slot 0) — optimistic; real entry ~slot 1-3 with MEV
- **Cost model**: none in Part-2 tables (gross); two-stage sim uses 100 bps buy + 100 bps sell
- **Max hold**: 1800 s (30 min), then forced exit at timeout
- **Source**: `eda/backtest/summaries_*.json`, `eda/two_stage/summary.json`

## Strategy Definitions

| ID | TP | SL | Exit Signal |
|---|---|---|---|
| `tp_2x_only` | 2× | none | — |
| `tp_2x_sl_50` | 2× | −50% | — |
| `trailing_30` | armed at 1.5×, trail −30% from peak | −60% flat | — |
| `deployer_sell_exit` | 3× | −50% | first `pump:sell` or `pump_amm:sell` action from deployer |
| `vol_stagnation_10` | 2× | −50% | vol < 10% of 60-s rolling max for 10 consecutive slots |
| `sell_pressure_5` | 2× | −50% | sell_vol > 1.5× buy_vol for 5 consecutive slots |

## Model Top-Decile Universe

| Strategy | n | mean ROI | median ROI | win % | med DD | med hold | exit reasons |
|---|---|---|---|---|---|---|---|
| tp_2x_only | 5000 | +7.33 | +0.094 | 52.2% | −11.8% | 35 s | tp 47%, timeout 53% |
| tp_2x_sl_50 | 5000 | +7.33 | +0.000 | 49.6% | −11.8% | 15 s | tp 45%, sl 24%, timeout 31% |
| **trailing_30** | 5000 | **+0.53** | **+0.104** | **60.6%** | ~0% | 26 s | trail 61%, sl 17%, timeout 22% |
| deployer_sell_exit | 5000 | +7.31 | −0.194 | 32.5% | −23.0% | 47 s | tp 25%, sl 30%, timeout 45% |
| vol_stagnation_10 | 5000 | +7.33 | +0.122 | 52.6% | −8.9% | 15 s | tp 42%, sl 23%, stag 11%, timeout 24% |
| **sell_pressure_5** | 5000 | **+7.32** | **+0.138** | **54.4%** | **−5.5%** | **12 s** | tp 41%, sl 21%, sp 23%, timeout 14% |

**Notes on mean ROI**: dominated by extreme right-tail winners. One 200× token can raise
the mean by 0.04 per trade across 5,000 positions. Mean is not the right optimisation
target — median and win rate are.

## Random Universe (baseline)

| Strategy | mean ROI | median ROI | win % |
|---|---|---|---|
| tp_2x_only | +0.068 | −0.013 | 30.2% |
| tp_2x_sl_50 | +0.067 | −0.019 | 29.2% |
| trailing_30 | +0.113 | 0.000 | 36.1% |
| deployer_sell_exit | +0.050 | −0.056 | 23.2% |
| vol_stagnation_10 | +0.068 | −0.011 | 30.6% |
| sell_pressure_5 | +0.081 | −0.001 | 34.0% |

## CEX Heuristic Universe (original task's "+20 if CEX" logic)

| Strategy | mean ROI | median ROI | win % |
|---|---|---|---|
| tp_2x_only | +0.074 | 0.000 | 30.9% |
| trailing_30 | +0.087 | 0.000 | 36.6% |
| sell_pressure_5 | +0.088 | 0.000 | 35.2% |

**CEX heuristic ≈ random universe.** Zero alpha from the "+20 if CEX" rule.

## Alpha Lift Summary

| Metric | Random | CEX | Model top-decile | Lift vs random |
|---|---|---|---|---|
| Win rate (trailing_30) | 36.1% | 36.6% | **60.6%** | **+24.5 pp** |
| Median ROI (trailing_30) | 0.000 | 0.000 | **+10.4%** | **significant** |
| Win rate (sell_pressure_5) | 34.0% | 35.2% | **54.4%** | **+20.4 pp** |

## End-to-End Two-Stage Simulation

Parameters: probe=0.1 SOL, scale=1.0 SOL, buy slip=100 bps, sell slip=100 bps,
buy_threshold=0.39, scale_threshold=0.85, abort_threshold=0.30, exit=trailing_30.

| Metric | Value |
|---|---|
| Candidates | 5,000 (subsample from 224,676 instant buys) |
| Probe-only | 3,336 (66.7%) |
| Aborted at 60 s | 824 (16.5%) |
| Scaled to full | 840 (16.8%) |
| Total PnL | +3,352 SOL |
| Mean PnL/trade | +0.67 SOL |
| Median PnL/trade | −0.012 SOL |
| Win rate | 27.6% |
| p10 PnL | −0.086 SOL |
| p90 PnL | +0.054 SOL |
| Worst | −0.84 SOL |
| Best | +3,410 SOL |

Tail-harvester profile: the right tail (rare 100×+ movers) fully dominates the mean.
The proper portfolio management approach is Kelly sizing on per-token expected value,
not fixed sizing.
