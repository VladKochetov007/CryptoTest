# Pump.fun Pre-Buy Scoring & Exit Strategy

End-to-end quant prototype on 500k Pump.fun tokens (2026-04-01 → 2026-04-20).

Solves both parts of the test task: instant buy/skip decision in <300 ms and post-launch exit strategies.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Dataset | 500k tokens, 19 days |
| Base hit-2x rate (30 min) | 16.0% |
| **Instant model OOF AUC** | **0.8014** (meta-LGBM, 77 features) |
| Baseline instant AUC | 0.7653 (47 features) |
| With-60s stage-2 AUC | 0.9452 |
| Top-decile win rate (trailing-30 exit) | **60.6%** vs 36.1% random |
| Two-stage PnL (5000 trades, 100 bps slippage) | **+3352 SOL** |
| Cross-target hypothesis (train 5x → predict 2x) | **Refuted** |
| Biggest discovery | `deployer_hr_7d` (IV=0.97, SHAP #1) |
| **End-to-end inference latency** | **6.1 ms** (293.9 ms budget remaining) |

---

## Best Combination for the Test Task

The test task goal is **"accuracy in selecting tokens with potential ROI > 2x after 30 minutes"** (Part 1) plus an **exit strategy** (Part 2). The combination that wins on both axes:

| Component | Choice | Why |
|---|---|---|
| **Model** | `meta__lgbm` (LightGBM, walk-forward CV) | OOF AUC **0.8014** — beats base 0.7653, XGB 0.7633, CatBoost 0.7642 |
| **Features** | 77 = 47 base + 30 meta-features | Adds multi-scale rolling deployer/funder/handle hit rates + text features |
| **Top feature** | `deployer_hr_7d` | IV=0.97, SHAP #1 — 7-day rolling win rate of the deployer wallet |
| **Entry rule** | score ≥ 39 → probe 0.1 SOL; ≥ 85 → 1.0 SOL after 60s re-score | Calibrated via isotonic regression to match base rate |
| **Exit strategy (single-stage)** | **`trailing_30`** | Win rate **60.6%**, median ROI **+10.4%**, capped drawdown -82% (vs -100% on TP-only) |
| **Exit strategy (two-stage)** | probe → re-score@60s → scale or abort + `trailing_30` | Total PnL **+3352 SOL** on 5000 trades with 100bps round-trip slippage |

**Lift attribution** vs random baseline on the same 5000-trade backtest:

| Source of edge | ΔWin rate | Δ Median ROI |
|---|---|---|
| Model selection (random → model top-decile) | +24.5 pp | +10.4 pp |
| Exit logic (TP-only → trailing_30) | +8.4 pp | +1.0 pp |
| Combined (random + TP-only → model + trailing_30) | **+30.4 pp** | **+11.7 pp** |

The model selection contributes ~3× more lift than the exit logic — **where you buy beats how you sell**.

![OOS equity curves](eda/plots/backtest_equity_curve_oos.png)

---

## What Was Built

### Part 1: Instant Decision (<300 ms)

A calibrated 0–100 score assigned at slot 0 (before any post-launch data). At deploy time the system reads from pre-computed rolling tables (deployer hit rates, funder stats, handle history) and runs a single LGBM inference in <10 ms.

**Measured inference latency (50k-iteration benchmark):**

| Step | Time |
|---|---|
| Deployer rolling stats lookup (dict) | 0.46 μs |
| Text features (regex + string ops) | 11 μs |
| Macro context lookup (dict) | 0.12 μs |
| **LGBM single-row predict** | **6,071 μs** ← bottleneck |
| Score + threshold | 0.15 μs |
| **Total** | **6.1 ms** |
| Budget remaining | **293.9 ms** (~98% headroom) |

LGBM predict dominates. No recursive rolling optimization needed — rolling stats are maintained as per-deployer deques (event-based, not a continuous time series), so each new token costs O(K) where K = tokens per deployer in the window (avg 5.4).

**Architecture:**
```
gRPC CreateEvent
  └─ 0.5 μs: lookup deployer_hr_7d / deployer_hr_24h from in-memory deque table
  └─ 11 μs:  text features (handle digit ratio, desc template score, name word count)
  └─ 0.1 μs: macro context (BTC price, SOL vol from last 5-min bar)
  └─ 6 ms:   LGBM inference (77 features, 278 trees) → raw probability
  └─ 0.1 μs: isotonic calibration → score 0–100
  └─ score ≥ 39 → probe buy (0.1 SOL)
  └─ score ≥ 85 → high-conviction (1.0 SOL, after 60s re-score)
```

**Why task heuristics fail** (all tested, all found weak or reversed):

| Task heuristic | Reality |
|---|---|
| CEX-funded → +20 | `is_cex` IV = 0.0001. Useless as boolean. CEX *name* matters: Gate.io +41% lift, MEXC -50% (anti-signal). |
| has_twitter → legitimacy | **Reversed**: no-twitter hit rate 18.4% vs with-twitter 14.8%. Bots add socials. |
| hasn't created tokens before → +10 | Non-monotone. Veterans with *wins* (`deployer_hr_7d`) are strongest signal. |
| deposit > 1 SOL → +20 | Continuous, not step. Works as a continuous feature, not threshold. |

### Part 2: Exit Strategies — Backtested on Historical Data

All results in `eda/backtest/` (JSON summaries committed) and `eda/two_stage/summary.json`.

#### Backtest Methodology

**Entry**: first slot price after deploy (slot 0 price, no slippage). In live trading, this is the price at which your buy lands before the first price move.

**Exit simulation**: slot-level replay using real `slot_features_60m.parquet` price data (one row per ~6.4s slot, up to 60 minutes). Each strategy is applied tick-by-tick to real prices — no curve-fitting, no look-ahead.

**Sizing (single-stage)**: flat 1 SOL per trade, no Kelly sizing, no compounding, no position scaling, no slippage. PnL = ROI × 1 SOL.

**Sizing (two-stage)**: 0.1 SOL probe at slot 0 → re-score using 60s post-launch data → scale to 1.0 SOL or abort. 100 bps buy + 100 bps sell slippage applied.

**Universe**: each run selects 5000 tokens (random seed 42 for reproducibility):
- `model_top` — top-decile by walk-forward OOF score (fully OOS, no look-ahead)
- `random` — random sample (no selection signal)
- `cex_heuristic` — tokens funded from a CEX wallet (task's suggested heuristic)

**What was tried (6 strategies)**:

| Strategy | Logic | What it tests |
|---|---|---|
| `tp_2x_only` | Sell at 2× entry price | Pure upside target, no downside protection |
| `tp_2x_sl_50` | 2× TP + -50% stop-loss | TP with hard floor |
| `trailing_30` | Arm at +50% peak, trail -30% from max | Lets winners run, exits on momentum reversal |
| `vol_stagnation_10` | Exit if volume < 10% of 60s rolling max for 10 slots | Sell into dying momentum |
| `sell_pressure_5` | Exit if sell_vol > 1.5× buy_vol for 5 consecutive slots | Order-flow flip detection |
| `deployer_sell_exit` | Exit on first deployer sell + 3× TP + -50% SL | Insider signal (hypothesis: rug warning) |

`deployer_sell_exit` was the worst (-19.4% median ROI): by the time the deployer sells, the rug has usually already happened. The signal is too late.

**Returns on capital (model_top + trailing_30, 19-day backtest, 5000 trades)**:

| Metric | Single-stage | Two-stage |
|---|---|---|
| Position size | 1 SOL flat | 0.1 SOL probe → 1.0 SOL scale |
| Slippage | None | 100 bps round-trip |
| Total invested | 5,000 SOL (across all trades) | 1,340 SOL (500 probes + 840 full) |
| Total PnL | +2,661 SOL raw / +2,022 SOL (ROI cap +500%) | +3,352 SOL |
| **ROI on invested** | **53.2% raw / 40.4% winsorized** | **250% on 1,340 SOL** |
| Avg concurrent positions | ~0.1 (median hold 26s, 321 trades/day) | ~0.1 |
| Working capital needed | ~1–2 SOL (sequential recycling) | ~1–2 SOL |
| Win rate | 60.6% | 27.6% (many probes aborted at 60s) |

> **Note on working capital**: with 321 trades/day and 26s median hold, average concurrency = 0.10 positions. In practice you never have more than 1–2 open at once, so ~2 SOL of working capital is enough to run all trades sequentially. The 5,000 SOL "total invested" figure assumes you treat each trade independently; with capital recycling the actual exposure is far smaller.

> **Note on raw vs winsorized**: a handful of tokens went 50–1000×. The raw PnL includes these tail events intact. The winsorized figure caps each trade at +500% per bet to show what you'd get without relying on extremely rare outliers.

Six exit strategies backtested on model top-decile, random, and CEX-heuristic universes (5000 tokens each):

| Strategy | Win % | Median ROI | Med hold | Notes |
|---|---|---|---|---|
| `trailing_30` | **60.6%** | **+10.4%** | 26s | Arms at +50%, trails -30% from peak |
| `sell_pressure_5` | 54.4% | +13.8% | 12s | Exit if sell_vol > 1.5× buy_vol for 5 slots |
| `vol_stagnation_10` | 52.6% | +12.2% | 15s | Exit if vol < 10% of 60s rolling max |
| `tp_2x_only` | 52.2% | +9.4% | 35s | Pure 2x take-profit |
| `tp_2x_sl_50` | 49.6% | 0.0% | 15s | 2x TP + -50% SL |
| `deployer_sell_exit` | 32.5% | **-19.4%** | 47s | Exit on deployer first sell — worst |

Random universe baseline: trailing_30 wins 36.1%, median ROI 0%.

**Two-stage integrated simulation** (`eda/two_stage/summary.json`):
- 3,336 probe-only (score too low at 60s → exit at 0.1 SOL)
- 824 abort at 60s (score dropped → cut the probe)
- 840 full scale (score held → 1.0 SOL)
- Result: **+3,352 SOL** gross PnL, 250% ROI on 1,340 SOL invested

**Backtests vs universe comparison:**

| Universe | Strategy | Win % | Median ROI |
|---|---|---|---|
| Model top-decile | trailing_30 | **60.6%** | **+10.4%** |
| Random baseline | trailing_30 | 36.1% | 0.0% |
| CEX heuristic | trailing_30 | 36.6% | 0.0% |

Alpha lift: **+24.5 pp win rate** from model selection vs random. CEX heuristic ≈ random (zero alpha).

### Part 3: Meta-Features Research (Overnight)

Multi-scale deployer rolling features built via cumsum+searchsorted (O(N log N), 12s for 500k tokens):

| Feature | IV | SHAP Rank |
|---|---|---|
| `deployer_hr_7d` — 7-day rolling win rate | **0.97** | **#1** (0.735) |
| `deployer_hr_24h` | 0.90 | #4 |
| `handle_hr_24h` — twitter handle track record | 0.25 | #8 |
| `funder_hr_7d` | 0.67 | #15 |
| `name_word_count` | 0.06 | #19 |

**Cross-target finding:** Training on harder label (hit_5x) does NOT improve hit_2x detection. The diagonal dominates the 4×4 cross-target AUC matrix. Actionable: use a separate model trained on hit_5x for the high-conviction 60s scale gate.

---

## Where to See Results

### Plots (human-readable)

All plots in `eda/plots/`:

**Backtest visuals (Part 2 — exit strategies):**

| File | What it shows |
|---|---|
| **`backtest_equity_curve_oos.png`** | **OOS equity over calendar time — 4 strategies × 3 universes + two-stage. Per-trade ROI winsorized at +500% to prevent tail outlier from swamping the comparison. The headline image.** |
| `backtest_winrate_by_universe.png` | Win rate × 6 strategies × 3 universes — visual proof model_top dominates |
| `backtest_median_roi_by_universe.png` | Median ROI × 6 strategies × 3 universes |
| `backtest_roi_distribution.png` | Per-trade ROI boxplot, model_top universe (clipped to [-100%, +500%]) |
| `backtest_hold_vs_roi.png` | Holding-time × ROI scatter for `trailing_30`, colored by exit reason |

**Model + feature visuals (Part 1):**

| File | What it shows |
|---|---|
| `meta_shap_importance.png` | Top-30 features by mean |SHAP| |
| `meta_feature_iv_bar.png` | Information Value of all 35 meta-features |
| `meta_deployer_scale_auc.png` | Univariate AUC at 1h/6h/24h/7d windows |
| `meta_text_auc.png` | Twitter handle + description + name feature AUCs |
| `meta_meme_kw_by_hour.png` | hit_2x rate for meme vs non-meme tokens by UTC hour |
| `cross_target_auc.png` | 4×4 AUC matrix — train on Nx, eval on Mx |
| `cross_target_lift.png` | 4×4 lift@top-10% matrix |
| `stack_distill_comparison.png` | Model AUC progression: base → meta → blend → distilled |
| `meta_handle_len_hitrate.png` | hit_2x rate by twitter handle length |
| `meta_desc_template_hitrate.png` | hit_2x rate by description template score |
| `shap_summary.png` | SHAP beeswarm (base model) |

### Scored Token Table

`eda/scoring/scored_1000_recent.csv` — 1000 most recent tokens with:
- `score_0_100`: calibrated score (0–100)
- `buy_decision`: score ≥ soft threshold
- `high_conviction`: score ≥ high-conviction threshold
- All features used for scoring

### EDA Notebook

`eda/notebook.py` — Marimo notebook with full statistical EDA. Run interactively with:
```bash
.venv/bin/marimo run eda/notebook.py
```
Or edit with:
```bash
.venv/bin/marimo edit eda/notebook.py
```

### Reports and Findings

- `eda/report.md` — full narrative report
- `eda/framing.md` — senior-quant framing of label, validation, feature design
- `eda/cross_target/summary.md` — cross-target experiment table
- `memories/` — deep-dives on every finding:

| Memory | Content |
|---|---|
| `00_overview.md` | Start here: key numbers, file map, run order |
| `01_leakage_audit.md` | What leaks, what was fixed |
| `02_significant_findings.md` | SHAP top-10, reversed-sign features, CEX vs boolean |
| `03_model_details.md` | Hyperparams, fold-by-fold AUCs, calibration thresholds |
| `04_backtest_results.md` | All 6 strategies × 3 universes |
| `05_open_issues_and_next_steps.md` | Next experiments, gRPC additions |
| `07_cross_target_table.md` | Full 4×4 cross-target matrix + analysis |
| `08_literature_features.md` | DeFi scam detection literature review |
| `09_meta_features_results.md` | Meta-feature IV, SHAP, new AUC 0.8014 |

---

## How to Run

```bash
# Setup
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # or install manually

# 1. Build base features (needs internet for BTC/SOL via ccxt)
.venv/bin/python eda/build_features.py      # ~3 min

# 2. Train base models (3 algos × 2 feature sets)
.venv/bin/python eda/train.py               # ~8 min, 16 cores

# 3. Build meta-features (~12s)
.venv/bin/python eda/build_meta_features.py

# 4. Train meta model + SHAP
.venv/bin/python eda/meta_train.py          # ~50s

# 5. Cross-target experiment (optional, slow)
.venv/bin/python eda/cross_target.py        # ~30 min

# 6. Calibration + 0-100 score + 1k CSV
.venv/bin/python eda/explain.py             # ~2 min

# 7. Exit strategy backtest
.venv/bin/python eda/backtest.py model_top  # ~2 min
.venv/bin/python eda/two_stage_sim.py       # ~1 min

# 8. Diagnostic plots
.venv/bin/python eda/meta_eda_plots.py      # ~1 min
.venv/bin/python eda/backtest_plots.py      # ~10s
.venv/bin/python eda/equity_curve_plot.py   # ~5s

# 9. Interactive EDA
.venv/bin/marimo run eda/notebook.py
```

---

## Data

Three parquet files (not in git — too large):

| File | Rows | Description |
|---|---|---|
| `tokens.parquet` | 500k | Token metadata, deployer info, twitter handle, description |
| `slot_features_60m.parquet` | 27.3M | Per-slot price, volume, holders, top_wallet_bought |
| `deployer_actions_60m.parquet` | 52.6M | Deployer on-chain actions (buy/sell/create/collect_fee) |

---

## Stack

Python 3.14 · Polars 1.40 · LightGBM 4.6 · XGBoost · CatBoost · DuckDB 1.5 · SHAP · scikit-learn · matplotlib · Marimo · ccxt (SOL/BTC OHLCV)
