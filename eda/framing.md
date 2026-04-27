# Pump.fun Pre-Buy + Exit — Statistical Framing

Source numbers: `eda/eda_report.json`. Plots: `eda/plots/` (only for visual confirmation of heavy-tail and survival; all decisions backed by stat tests below).

## 1. Dataset facts

| Quantity | Value |
|---|---|
| tokens | 500,000 |
| unique deployers | 91,765 |
| top single deployer (factory) | 8,465 tokens |
| tokens with any 60-min trade | 470,394 (94.1%) |
| tokens with first-slot price | 470,256 (94.0%) |
| graduation rate (ATH ≥ $69k) | **0.48%** |
| ATH mcap p50 / p99 | $4.3k / $37k |

## 2. Label design

Target chosen: **`hit_2x_30m`** = `pmax(0..1800s) / first_slot_price >= 2`.

| Label | Base rate | Notes |
|---|---|---|
| hit_1.5x_30m | 26.1% | weak |
| **hit_2x_30m** | **16.0%** | balanced, ML-friendly |
| hit_3x_30m | 7.4% | sparse |
| hit_5x_30m | 3.4% | rare |
| hit_grad ($69k mcap) | 0.48% | needs different model |

ROI distribution (30 min): p50=1.18, p90=2.59, p99=11.3. Most tokens drift sideways or down.

**Anchor caveat**: anchoring at first-slot price gives an attainable label *for a deployer-time bot* but a real bot pays a higher entry (after 1–2 slots of MEV). For real PnL evaluation, also compute label anchored at 5-second post-deploy mid-price.

## 3. Time stability — REGIME RISK

Split data into 10 chronological chunks:

| Window slice | hit_2x rate | hit_grad rate |
|---|---|---|
| chunk 0 (oldest) | 12.4% | 0.38% |
| chunk 4 | 19.0% | 0.43% |
| chunk 9 (newest) | 18.1% | 0.63% |

Range: hit_2x 12.4–19.0% (~50% relative drift), hit_grad 0.35–0.63%. **Static train/test split would lie. Walk-forward CV is mandatory.**

## 4. Univariate feature analysis (Part 1, deploy-time only)

### 4.1 Categorical — Information Value (IV) and Cramér's V

| Feature | IV | Cramér V | χ² p | Interpretation |
|---|---|---|---|---|
| `deployer_wallet_source_cex_name` | **0.0778** | 0.098 | 3e-30 | Identity matters, not the boolean. Gate.io 22.5% (lift 1.41), MEXC 8.0% (lift 0.50!), Binance 14.6%. |
| `has_twitter` | 0.016 | 0.047 | <1e-200 | **Reversed**: no-twitter rate 18.4% > with-twitter 14.8%. Bots add socials, real degens don't. |
| `has_desc` | 0.013 | 0.041 | <1e-150 | Reversed (17.6% no-desc vs 14.6% has-desc). |
| `has_website` | 0.006 | 0.028 | <1e-80 | Reversed but weak. |
| `has_image` | 0.004 | 0.023 | <1e-50 | Forward but weak. |
| `is_cex` | **0.0001** | 0.004 | 0.008 | **USELESS**. Reject the task-hint heuristic. |

Headline: every "social presence" rule from the task description is weak or reversed in this dataset. The +20 / +15 heuristics are wishful thinking. Use the CEX-name level as a categorical with WoE encoding (different per exchange — Gate.io is bullish, MEXC is bearish).

### 4.2 Continuous — AUC, Spearman ρ, KS

| Feature | AUC | Spearman ρ | KS | Notes |
|---|---|---|---|---|
| `deployer_wallet_balance_after_sol` | 0.563 | +0.080 | 0.110 | best deploy-time signal |
| `deployer_deposit_amount` | 0.550 | +0.063 | 0.108 | monotonic decile lift |
| `deployer_wallet_source_amount_sol` | 0.549 | +0.063 | 0.108 | ρ = 0.9996 with deposit → drop one |
| `deployer_wallet_balance_before` | 0.549 | +0.062 | 0.085 | |
| `same_ticker_today` | 0.520 | +0.026 | 0.063 | clone proxy |
| `image_hash_seen_total` | 0.519 | +0.025 | 0.058 | clone proxy |
| `name_len` | 0.447 | −0.067 | 0.076 | **short names win**, reversed |
| `deployer_prior_n` | 0.484 | −0.020 | 0.088 | **factory output ≠ winning** |
| `ticker_len` | 0.498 | −0.003 | 0.012 | noise |

No single deploy-time continuous feature exceeds AUC 0.57. Combined ensemble realistically reaches AUC 0.60–0.66 (estimated from joint Cramér's V envelope).

### 4.3 Multicollinearity (drop list before fitting)

| Pair | Spearman ρ | Action |
|---|---|---|
| deposit_amount ↔ source_amount_sol | 0.9996 | drop one |
| balance_before ↔ balance_after_sol | 0.824 | drop one (or use diff) |
| deployer_prior_n ↔ balance_before | 0.519 | accepted, low VIF after drop |
| image_hash_seen_total ↔ same_ticker_today | 0.523 | combine into `clone_score` |

## 5. Post-buy lookback features (60s window) — supports 2-stage decision

Tested AUC if you allow yourself 60s of post-launch observation before final size:

| Feature | AUC |
|---|---|
| `buy_vol_60s` | **0.819** |
| `holders_60s` | 0.800 |
| `sell_vol_60s` | 0.791 |
| `first_buy_vol_sol` | 0.672 |
| `top_wallet_60s` | 0.602 |

Implication: pre-buy at slot 0 is statistically hard. A two-stage policy (small probe at slot 0, decision-to-add or exit at 30–60s based on micro-features) recovers a lot of edge. Recommend the system propose **two scores**: instant_score (deploy-time only) and confirm_score (60s lookback).

## 6. Exit signals — descriptive

Deployer first-sell action distribution (211,231 / 470k tokens have a deployer sell in the first 60min):

| Quantile | seconds since deploy |
|---|---|
| p10 | 1 s |
| p25 | 3 s |
| p50 | 21 s |
| p90 | 899 s |
| share ≤ 30 s | **55.1%** |
| share ≤ 120 s | 72.1% |

Translation: in 45% of tokens the deployer dumps within 60 min, and more than half of those happens in the first 30 seconds. The "deployer sells" rule is essentially a frequentist rug detector with a clean 30-second window.

Creator-fee claim (a graduation-correlated PumpFun action) appears in 159k tokens, p50 ≈ 528 s — much later, so it is information for hold-until-grad strategies, not for early exits.

## 7. Modeling plan

### Part 1 — pre-buy scoring (≤300 ms latency budget)

1. **Feature set (deploy-time only, no slot data)**: `deployer_deposit_amount` (log1p), `deployer_wallet_balance_after_sol` (log1p), `deployer_prior_n` (log1p), `clone_score` = `image_hash_seen_total + same_ticker_today`, `name_len`, `has_image`, `has_twitter` (reverse-encode), `has_desc` (reverse-encode), `cex_name_woe` (target-encoded with smoothing on training fold only).
2. **Model**: gradient boosting (LightGBM or XGBoost) with monotonic constraints where direction is established. Fall back to logistic regression with WoE bins for explainability and to prove the GBT isn't overfitting.
3. **Validation**: walk-forward (10 folds, expanding window). Within each fold compute target-encoding from train only. Report PR-AUC + expected ROI at score thresholds.
4. **Calibration**: isotonic on the validation fold, then map to the 0–100 score requested in the task as the calibrated probability × 100.
5. **Decision rule**: buy if `score ≥ τ` where τ is chosen to maximize expected log-return given assumed entry-execution slippage and gas cost; sweep τ over grid.

### Part 2 — exit strategy

Sample 100+ "buys" (positive-score tokens at the deploy moment) and back-test on `slot_features_60m`:

| Strategy | Rule |
|---|---|
| TP_2x | exit when price ≥ 2 × entry |
| TP_2x_SL_50 | TP 2× or SL −50% |
| Trailing 30% | trailing stop on the running max |
| Volume stagnation | exit if `volume_sol` < 10% of 60-s rolling max for 10 consecutive slots |
| Deployer sells | exit on first row in `acts` table matching `pump:sell` or `pump_amm:sell` |
| Top-wallet exit | exit if `top_wallet_bought=False` while sells dominate (sell_volume > 1.5 × buy_volume) for 5 slots |

Backtest using `vectorbt` or a polars-native event simulator. Report per strategy: median ROI, mean ROI, Sortino, max DD per token, hold time p50/p90, hit ratio (≥ 1×), full return distribution. Combine winning rules via OR (first exit triggered) → expect material improvement vs static 2× TP.

### Validation discipline

- **No leakage**: drop `latest_market_cap_usd`, `ath_market_cap_usd` from feature set (label-only).
- **Walk-forward** with embargo of 1 hour between train end and test start.
- **Deployer-grouped split** to avoid the same factory wallet appearing in train and test, since 50% of the volume is from a small set of deployers.
- **Sanity baselines**: random selection (16% hit_2x base), CEX-naive heuristic (≈ same), price > X SOL threshold. ML must beat all three by ≥3× lift in the top decile.

## 8. Improvements (Part 3 in task)

1. **Add to gRPC stream**: deployer's last 24 h tokens win-rate, deployer balance flow graph (CEX → wallet hops), bonding curve initial buy size in same tx, holder Gini at slot 0, MEV bot tx in same block.
2. **ML stage**: LightGBM with monotone constraints for Part 1; for exits, a small recurrent or boosted classifier on the 60-s slot window predicting "max return in next 5 slots > entry".
3. **Adversarial filtering**: known sybil/bundler clusters (use `sybil-detection` skill) — filter before scoring.
4. **Latency**: precompute deployer-state features in a streaming key-value store, keyed by deployer address, updated on every gRPC event. Lookup at deploy event is then O(1).

## 9. Open questions for user

- Confirm the entry-execution model — slot-0 mid-price, or slot-0 ask, or 1-slot delay?
- Is there a max position size (sets the relevant ROI scale and changes thresholds)?
- Cost model: gas + slippage + fees — defaults are 0.5–1.5% in SOL on tiny PumpFun trades; need a confirmed number to set τ.
- Acceptable latency for Part 2 exit decisions — do we need on-chain executor latency or off-chain backtest only?
