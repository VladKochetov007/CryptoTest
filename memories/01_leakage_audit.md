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

## with60s AUC Discrepancy

- With `price_max_60s` included: OOF AUC = 0.99 (suspicious)
- After removing `price_max_60s/min_60s`: OOF AUC = 0.945 (plausible, not tautological)

Root cause: if a token's price hits 2× within 60 s, `price_max_60s ≥ 2×p0` and
`label = True` simultaneously. Model learns `price_max_60s / p0 ≥ 2 → predict 1`
with near-perfect accuracy — essentially computing the label directly.
