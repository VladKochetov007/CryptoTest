# Leakage Audit

## Confirmed Leaky → Excluded from Features

| Column | Reason |
|---|---|
| `ath_market_cap_usd` | Future maximum — direct label proxy |
| `latest_market_cap_usd` | Snapshot taken after token lifecycle |
| `pmax_30m` (computed) | IS the label numerator |
| `peak_sec_30m` (computed) | Derived from future prices |
| `price_max_60s` | If 2× hits within 60 s → label=True by construction (AUC inflated 0.95→0.99 when included) |
| `price_min_60s` | Same issue, less severe |

## Confirmed Clean — Deploy-Time Only

These are available in the gRPC event at `pump:create_v2` emission:
- `deployer_deposit_amount`, `deployer_wallet_balance_before/after`
- `deployer_wallet_source_amount_sol`, `is_cex`, `deployer_wallet_source_cex_name`
- Metadata booleans/lengths
- Clock features (UTC hour, NY/London/Tokyo hour, DOW, sin/cos phase)
- SOL/BTC OHLCV from Binance at deploy time (5-min bar ending before deploy)

## Subtle Leakage Risks (fixed)

### Deployer History Aggregates
**Risk**: `deployer_prior_grad` could include future tokens from same deployer.  
**Fix**: DuckDB window function with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` — each token sees only strictly earlier deployments from the same wallet.

### Cross-Section Regime
**Risk**: `hit20k_rate_prev_60m` uses current-minute bucket.  
**Fix**: 1-minute rolling window then `shift(1)` — token at minute M sees windows ending at M-1.

### Ticker / Image Clone Counts
**Risk**: `same_ticker_today` includes same-day later tokens.  
**Fix**: `ticker_clash` CTE uses `t2.deploy_time_unix < t.deploy_time_unix` strict inequality.

### Funder History
**Risk**: Source-wallet history includes later fundings.  
**Fix**: Same window-function approach as deployer history.

## Post-Deploy Features (60 s) — Leakage-Adjacent Note

`buy_vol_60s`, `sell_vol_60s`, `holders_60s`, `top_wallet_60s`, `trades_60s` are
post-launch. They do NOT directly encode the label (label uses t=0..1800, first 60 s is
a strict subset). However they are NOT available at the ≤300 ms decision moment.

Appropriate use: stage-2 scaling decision after 60-s observation window.

## Tie-Time Label Leak in Rolling Helpers (FIXED 2026-04-28)

`compute_rolling_group_features()` originally used `cum_hits[i] = Σ hits[0..i-1]` as
the "prior" running sum after stable sort by time. When two tokens by the same
deployer share `deploy_time_unix` (~18,400 such pairs in 500k), the second peer
read a `cum_hits` that already incorporated the first peer's `hit_2x` label.
Same-second peers thus saw each other's outcomes → direct label leakage in
`deployer_hr_24h`, `deployer_hr_7d`, `funder_hr_*`, `handle_hr_24h`.

**Fix:** index `cum_hits[hi[i]]` where `hi = searchsorted(times, times, side="left")`.
For tied positions all peers map to the same `hi` (first occurrence of the tied
time), so prior-hit counts strictly exclude both self AND tied peers.

**AUC delta:** meta__lgbm 0.8014 (leaky) → 0.8007 (clean). Tiny because tie pairs
are 3.7% of rows and the contaminated information is small. But the fix is
correctness-critical — `deployer_hr_7d` is the top SHAP feature and any leak
there inflates calibration.

**Test:** `eda/leakage_tests.py` t4 verifies tie-peers get identical priors on a
random sample of 50 dup-pairs every retrain.

## Embargo (FIXED 2026-04-28)

`train.walkforward_indices(embargo=0)` in production code despite the docstring
claiming "Embargo 1 hour". Hit_2x label window ends at deploy + 1800 s, so a
token at the fold boundary's right edge had its label observed during fold k+1's
training. Switched to seconds-based embargo (default 3600 s, i.e. 2× the label
window). Same fix applied to `meta_train.walkforward_splits()`.

**AUC delta:** instant/lgbm 0.7653 → 0.7716 (slight rise — fold-1 with 1.4 days
training data was previously dragging the OOF mean down; the new schedule
shrinks fold-1 further but the macro signal stabilises). meta__lgbm 0.8014 →
0.8007.

## with60s AUC Discrepancy

- With `price_max_60s` included: OOF AUC = 0.99 (suspicious)
- After removing `price_max_60s/min_60s`: OOF AUC = 0.945 (plausible, not tautological)

Root cause: if a token's price hits 2× within 60 s, `price_max_60s ≥ 2×p0` and
`label = True` simultaneously. Model learns `price_max_60s / p0 ≥ 2 → predict 1`
with near-perfect accuracy — essentially computing the label directly.
