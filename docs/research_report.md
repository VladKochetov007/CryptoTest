# Pump.fun Pre-Buy Scoring — Research Report

**Objective**: score a new pump.fun token within milliseconds of its creation event, before any price action, and decide whether to buy. The target is `hit_2x`: the token's market-cap doubled within the first hour after launch.

---

## 1. Problem framing

Pump.fun launches roughly 50 000 tokens per day. ~85% go to zero within minutes. The pre-buy window is 200–400 ms after the `CreateEvent` is emitted on Solana — before bonding-curve buyers arrive. At that point only on-chain metadata and wallet history are available; no price data exists.

**Label**: `hit_2x = 1` if `ath_market_cap_usd >= 2 × initial_market_cap_usd` within the first 60 minutes.
`initial_market_cap_usd` is approximated from the genesis bonding-curve price × total supply.

**Base rate**: 15.2% (varies 12.4%–19.0% across time folds — regime-sensitive).

**Dataset**: 500 000 tokens, ~75 000 with `hit_2x` label assigned (remainder have no slot-feature data).

---

## 2. Data sources

| File | Contents |
|---|---|
| `tokens.parquet` | On-chain token metadata: name, ticker, description, image hash, deployer address, funder wallet, CEX name, website, twitter handle, telegram |
| `eda/features.parquet` | Per-token derived features: deployer wallet SOL balance, deployer history (alltime), funder history, market activity at deploy time, clock features, 5-min Binance macro bars joined at deploy timestamp |
| `slot_features_60m.parquet` | 1-minute OHLC price-in-SOL per token for 60 minutes post-deploy. Used for backtest only — never for training features. |
| `eda/meta_features.parquet` | Multi-scale rolling features computed from `features.parquet` + `tokens.parquet` |

---

## 3. Leakage bugs found and fixed

### 3.1 Price max as feature (`price_max_60s`)

An early feature used `price_max_60s` — the maximum price in the first 60 seconds — as a predictor of `hit_2x`. This is post-deploy price data that is unavailable at the pre-buy decision point. **Removed**.

### 3.2 ATH market cap as feature

`ath_market_cap_usd` is the ground-truth label source. Using it or any monotone transform of it as a feature is direct label leakage. **Removed**. Now only used to construct `hit_2x`.

### 3.3 Embargo = 0 in walk-forward splits

`train.py` defaulted `embargo=0`. The `hit_2x` label window is 60 minutes. Without embargo, tokens at the boundary of the train/validation split could have their label window overlap with adjacent training tokens' price data. Fixed: `embargo_sec=3600` (1 hour). AUC drop was < 1 pp — leakage was mild.

### 3.4 Tie-time label leakage in rolling group features (critical)

`compute_rolling_group_features` uses `np.cumsum + np.searchsorted` per group. Original code used `cum_hits[i]` (self-inclusive) as the upper bound and `side="right"` for `hi`. When two tokens by the same deployer share the same `deploy_time_unix`, token at sorted index `i` would read `cum_hits[i]` = Σ hits[0..i-1] **plus hits[i-1]** (the peer's hit). This affected `deployer_hr_7d` (top-1 SHAP feature), `funder_hr_*`, and `handle_hr_24h`.

Fix: `hi = np.searchsorted(times, times, side="left")` — first index where `times[j] >= times[i]`. This strictly excludes self AND all same-second peers. `cum_full[hi[i]]` reads the cumsum up to (not including) the tie group. OOF AUC after fix: 0.7977 (was 0.8007 — the 3 bps drop is the fair cost of removing the leak).

### 3.5 Non-stationary price levels as features (`btc_close`, `sol_close`)

Both are I(1) processes. A model trained on historical BTC close = $45k will extrapolate incorrectly at $75k. These appeared in early SHAP beeswarm plots with a strong directional effect that was clearly encoding bull-market regime, not a cross-sectional signal.

**Fix**: replace with stationary alternatives:
- `sol_vol_1h`, `sol_vol_24h` — realized vol (std of 5-min log-returns) — already stationary
- `sol_ret_1h`, `sol_ret_24h`, `btc_ret_1h` — percentage returns only
- `btc_ret_24h` — derived via join_asof on deploy timestamps
- `sol_vol_ratio = sol_vol_1h / sol_vol_24h` — short/long vol ratio (regime signal)
- `sol_btc_ret_spread = sol_ret_1h - btc_ret_1h` — SOL excess return

---

## 4. Models tested

### 4.1 Instant — features available at deploy time only

| Model | OOF AUC | PR-AUC | Notes |
|---|---|---|---|
| LightGBM | 0.7628 | 0.422 | Best instant model |
| XGBoost | 0.7628 | 0.424 | Essentially tied with LGBM |
| CatBoost | 0.7616 | 0.419 | Slightly weaker |

All three gradient boosted trees perform nearly identically on the instant feature set. Hyperparameters are lightly tuned (grid search over `num_leaves`, `learning_rate`, `feature_fraction`).

### 4.2 With60s — adds first 60 seconds of price/volume data

| Model | OOF AUC | PR-AUC | Notes |
|---|---|---|---|
| LightGBM | 0.9453 | 0.808 | Huge lift from 60s of price action |
| XGBoost | 0.9448 | 0.807 | Near-identical |
| CatBoost | 0.9434 | 0.798 | Slightly below |

AUC jump from ~0.76 to ~0.945 confirms that price action in the first minute is extremely informative. This model is used as a secondary filter in the two-stage simulation and as an exit signal in `backtest_model_exit.py`.

### 4.3 Meta — base + multi-scale rolling window features

| Model | OOF AUC | PR-AUC | Notes |
|---|---|---|---|
| LightGBM | **0.7977** | 0.457 | +324 bps over instant baseline |

The meta model is the **production model** for pre-buy scoring. It adds 43 meta features on top of the 41 base features, computed via O(N log N) rolling window group aggregations. No price data used.

**Fold AUCs**: 0.8277, 0.7540, 0.7802, 0.8117, 0.8058 — fold 2 is notably weaker (regime shift in that time window).

---

## 5. Feature engineering

### 5.1 What worked

**Deployer hit rate (7d window)** — `deployer_hr_7d` — top feature by SHAP (0.693). Serial winners are persistent. A deployer who consistently launches successful tokens keeps doing so; the 7-day window captures recent behavior without overfitting to stale history.

**Time since last deploy** — `deployer_seconds_since_last` — SHAP 0.381, rank 2. Short gaps indicate serial spam; very long gaps suggest infrequent, intentional launches. Non-linear relationship well-captured by LGBM.

**Deployer wallet balance** — `deployer_wallet_balance_after_sol` (SHAP 0.157) and `deployer_deposit_amount` (0.133). Deployers who commit more SOL tend to be genuine. The deposit amount is the SOL credited to the bonding curve at creation.

**Name uppercase chars** — `name_upper_chars` — SHAP 0.117. Surprising rank-6 signal. Appears to encode deployer persona styles that correlate with quality.

**Twitter handle hit rate** — `handle_hr_24h` — SHAP 0.101. Handles associated with recent successful launches are predictive.

**Image hash prior count** — `image_hash_prior_count` — SHAP 0.060. Reused images (SHA256 exact match) flag serial deployers recycling assets. First-use (0) is ambiguous; count > 0 is a negative signal.

**Funder 7d hit rate** — `funder_hr_7d` — SHAP 0.054. Smart-money funder wallets (those whose deployers historically succeed) carry signal.

**Meme keyword rolling win rate** — `meme_kw_hr_24h` — SHAP 0.049. Regime signal: when meme-keyword tokens are running hot in the last 24h, new ones are more likely to hit. Computed via 15-min bucket aggregation + shift(1) to avoid leakage within the bucket.

**Mint address ends in "pump"** — `mint_suffix_pump` — SHAP 0.059. Genuine pump.fun-generated mints always end in "pump". Some deployers use vanity mints (custom suffix) — that's a behavioral difference.

### 5.2 What failed or was weak

**`is_cex`** (binary) — near-zero IV. The categorical `deployer_wallet_source_cex_name` carries signal that the boolean masks. CEX source name encodes deployer sophistication: Gate.io-funded deployers had the highest lift (1.41×), MEXC +0.50×, others near neutral. But as a standalone feature, `is_cex` alone is not useful.

**Meme keywords** (`name_has_meme_kw`, `ticker_has_meme_kw`) — near-zero IV. "Doge", "pepe", "moon" appear in >40% of tokens — too common to be selective. Only the rolling win rate for the category (`meme_kw_hr_24h`) provides signal.

**Boolean metadata features** (`has_image`, `has_website`, `has_telegram`, `has_desc`, `has_twitter`) — replaced by continuous analogs. `has_image` → `image_hash_prior_count`; `has_desc` → `desc_len`; `has_twitter` → `handle_len`. The boolean survives only as `has_image` (weak but non-zero SHAP 0.059).

**Description template phrases** — `desc_has_deployed_template`, `desc_template_score` — near-zero individually. Most spam tokens use the same template but so do legitimate ones. Very low information value.

**Clock features** — `utc_hour`, `ny_hour`, etc. — low SHAP, no clear diurnal pattern in the data after controlling for deployer quality. The cyclic encoding (sin/cos) is correct but the underlying signal is weak.

**`has_twitter` reversed sign** — discovered in early EDA: tokens with a twitter handle had *lower* hit rates in the raw data. This was a Simpson's paradox: high-activity spam campaigns often include handles. After controlling for `handle_hr_24h`, the direct effect of presence is near-zero.

### 5.3 Feature design principles applied

- All price-level features removed (non-stationary I(1)). Only volatility (realized) and percentage returns used.
- Every boolean replaced by a continuous count or rolling rate where possible.
- Cross-asset features as spreads/ratios (`sol_btc_ret_spread`, `sol_vol_ratio`) rather than levels.
- Rolling features use `side="left"` searchsorted to guarantee strict past-only (tie-safe).
- Image similarity: SHA256 exact hash breaks on 1-pixel change. Perceptual hashing (pHash + Hamming distance) would be more robust but requires downloading images — left as future work.

---

## 6. Exit strategies tested

All strategies were backtested on the `model_top` universe (top-10% OOF score, 5000 tokens), 30-minute window from first slot observation, gross ROI (no cost).

| Strategy | Win rate | Mean ROI | Median ROI | Total PnL (SOL) | Notes |
|---|---|---|---|---|---|
| `tp_2x_only` | 0.550 | 23.3× | 1.000 | dominated by tail | Waits for 2× TP; 49.5% timeout |
| `tp_2x_sl_50` | 0.526 | 23.3× | 0.311 | — | Adds 50% hard SL |
| `trailing_30` | 0.631 | 0.642 | 0.142 | 2184 SOL | Trail 30% from cum-max after 1.5× arm |
| `deployer_sell_exit` | 0.343 | 26.7× | −0.205 | — | Exit on deployer sell signal; mean driven by 3× TP outliers |
| `vol_stagnation_10` | 0.551 | 23.2× | 0.346 | — | Exit on 10-slot zero-volume stagnation |
| `sell_pressure_5` | 0.567 | 23.3× | 0.311 | — | Exit on 5+ consecutive sell-pressure slots |

**`tp_2x_only` and siblings**: mean ROI of 23× is misleading — it's dominated by a handful of 100× winners. Median ROI is ~0.31, meaning the typical trade barely breaks even. Not robust.

**`trailing_30`**: best risk-adjusted result. Arm at 1.5× entry, trail 30% drawdown from cum-max, hard SL at −60%. Median ROI 0.142, win rate 0.631. This is the chosen strategy.

**`deployer_sell_exit`**: exits when the deployer wallet sells on-chain. Win rate only 0.34 — deployer sells often happen before the price peak but the signal is too noisy. High mean is entirely from 3× TP outliers.

### 6.1 Trailing parameter sweep

Grid: `arm_mult ∈ {1.3, 1.5, 2.0}` × `trail_frac ∈ {0.8, 0.7, 0.6, 0.5, 0.4}` (drawdown tolerance = 1 - trail_frac).

Best by total PnL (5000 model-top tokens, gross):

| arm | trail | drawdown | total PnL | win rate |
|---|---|---|---|---|
| 1.5× | 0.8 | 20% | 2202 SOL | 0.679 |
| 1.3× | 0.7 | 30% | 2184 SOL | 0.613 |
| 1.3× | 0.8 | 20% | 2183 SOL | 0.708 |
| 1.5× | 0.7 | 30% | 2165 SOL | 0.631 |

Tight trailing (20% drawdown) is marginally better. The 1.5× arm at 20% drawdown is the sweep winner. Production uses 30% drawdown for robustness to wide spreads at low liquidity.

### 6.2 Model-as-exit (second-stage scoring)

The `with60s` model (AUC 0.945) is applied at t = 60s as an early-exit trigger. If the 60s score falls below 0.30, the position is exited with 100 bps slip. Otherwise the trailing stop runs.

Result: model-as-exit total PnL = 2126 SOL vs trailing-only baseline 2183 SOL (−57 SOL, −2.6%). The `with60s` signal adds 388 early exits but at a net loss — it cuts some losers but also cuts winners that had not yet armed the trailing stop. Net effect is slightly negative. **Not adopted in production.**

---

## 7. Liquidity and slippage model

### 7.1 Pump.fun bonding curve mechanics

Pump.fun uses a **virtual constant-product AMM**: `k = v_sol × v_token`.

Initial parameters at genesis:
- `v_sol_init = 30 SOL` (30 000 000 000 lamports)
- `v_token_init = 1.073 × 10⁹ tokens` (including virtual reserve of 280M tokens never withdrawable)
- 1% fee on every buy (deducted from SOL input **before** the curve) and every sell (deducted from SOL output **after** the curve)

Spot price at genesis: `30 / 1.073×10⁹ ≈ 2.8×10⁻⁸ SOL/token`

Price impact for a 0.1 SOL buy at genesis:
```
tokens_out = v_token - k / (v_sol + 0.1 SOL) ≈ 3.56M tokens
exec_price ≈ 0.1 / 3.56M ≈ 2.81×10⁻⁸ — impact ≈ 0.17%
```

For a 1 SOL buy at genesis: impact ≈ 1.7%.

**Graduation**: when `real_sol_reserves ≥ 85 SOL`, the curve closes and the token migrates to PumpSwap (or Raydium for pre-March 2025 tokens). Only ~1.4% of tokens graduate.

### 7.2 Transaction cost model used in backtest

**Flat-slip model** (used in `eda/backtest.py` and `eda/pipeline.py`):
- `ENTRY_SLIP = 1.3%` — 1% pump fee + 0.3% AMM impact for a ~0.1 SOL probe
- `EXIT_SLIP = 1.3%` — same on the sell side
- **Roundtrip cost: 2.6%** per trade

**AMM-fill model** (used in `eda/two_stage_sim.py`):
- Entry: exact bonding-curve fill using `buy_tokens(v_sol, v_token, real_token, sol_in)` with slot-0 reserves
- Exit: `sell_tokens(v_sol_post, v_token_post, real_sol_post, tokens_in)` at exit-slot reserves
- Two-stage logic: 0.1 SOL probe at t=0 → score with `with60s` model at t=60s → if `p ≥ 0.85`, scale to 1 SOL full position; if `p < 0.30`, abort

### 7.3 AMM simulation results

On an **unfiltered** 20 000-token universe:
- Total PnL: **−68.3 SOL** (negative — fees dominate on a random universe)
- Win rate: 29%
- 18.3% full positions, 16.3% abort at 60s, 65.3% probe only

This confirms the model filter is load-bearing: buying anything is a losing strategy at 2.6% roundtrip cost on a 15% base-rate signal.

On the `model_top` universe the flat-slip backtest shows 2184 SOL total gross PnL, ≈ 2184 − 0.026 × 5000 = **1054 SOL net** (rough estimate; exact depends on position sizing).

---

## 8. Two-stage pipeline architecture

```
CreateEvent (t = 0ms)
    ↓
Fetch deployer history, funder wallet, token metadata from RPC / local cache
    ↓
Compute 84 meta features (target: < 150ms)
    ↓
meta__lgbm.predict() → p_pre  (target: < 5ms)
    ↓
if p_pre >= threshold (top 10%):
    buy 0.1 SOL probe
    ↓
    wait 60 seconds
    ↓
    observe price, volume, buyer count
    ↓
    with60s.predict() → p_post
    ↓
    if p_post >= 0.85: scale to full 1 SOL position, run trailing stop
    if p_post < 0.30:  exit immediately
    else:              hold probe through trailing stop
```

The live pipeline requires Yellowstone Geyser gRPC subscription for sub-200ms event delivery. See `.claude/skills/yellowstone-grpc/SKILL.md` for the streaming architecture.

---

## 9. Permutation null test

Ran with shuffled `hit_2x` labels within each fold. Expected AUC ∈ [0.48, 0.52]. Actual permuted OOF AUC: 0.499 ± 0.006 across 5 shuffle runs. Confirms no leakage path that survives label destruction. (After the tie-time fix in §3.4.)

---

## 10. Regime sensitivity

The model is trained on a dataset spanning multiple market regimes. Base rate variation:
- Fold 1 (earliest): 12.4% — bear/quiet period
- Fold 5 (latest): 19.0% — active period

This 6.6 pp swing is significant. `hit20k_rate_prev_60m` (platform-wide regime feature) captures some of this but the model is not explicitly regime-conditioned. Consider time-decayed sample weights or rolling re-training in production.

---

## 11. Open issues and future work

| Issue | Status | Notes |
|---|---|---|
| Entry price at slot 0 vs slot 1 | Open | Buying at `price_sol_per_token[0]` assumes instant fill. In reality the first filled slot may be 1–5 seconds after `CreateEvent`. A 5s entry relabeling experiment is pending. |
| Holder Gini coefficient | Not implemented | Requires `getTokenLargestAccounts` via Helius — network call, not in offline dataset. |
| Perceptual image hashing | Not implemented | SHA256 exact match breaks on 1-pixel edits. pHash + Hamming distance would give a continuous similarity score. |
| Kelly fraction sizing | Not implemented | Requires calibrated probability estimates. Current predictions are not well-calibrated (Brier score 0.192 vs 0.15 theoretical). |
| Soft-label distillation | Not implemented | Train `hit_2x` student with `hit_5x` teacher signal. `hit_5x` base rate ~4%, much cleaner label. |
| Mempool priority fee | Not implemented | High priority fees at deploy time may indicate coordinated launch (sniper-bots paying for early fill). |
| Sniper count at slot 0–3 | Not implemented | Count distinct buyer addresses in first 3 slots. Coordinated bundle signal. |
| Rolling re-training | Not implemented | Model trained once on full history. In production, rolling 30-day re-train would adapt to regime shifts. |
