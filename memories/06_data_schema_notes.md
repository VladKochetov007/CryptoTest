# Data Schema Notes

## Dataset: 500k Pump.fun tokens (2026-04-01 → 2026-04-20)

### `tokens.parquet` (500k rows, 25 cols)

Key fields for modeling:
- `token_id` INT — join key across all tables
- `deploy_time_unix` BIGINT — sort key for walk-forward CV
- `deployer_address` VARCHAR — group key for grouped CV
- `deployer_deposit_amount` DOUBLE — SOL funded to deployer; NULL for 21,069 rows (4.2%)
- `deployer_wallet_source` VARCHAR — same null rate as deposit_amount
- `deployer_wallet_source_is_cex` BOOL — True for 17,720 rows (3.5%)
- `image_hash_sha256` VARCHAR — NULL for 79,331 rows (15.9%)
- `ath_market_cap_usd` DOUBLE — **LEAKY**, label-only
- `latest_market_cap_usd` DOUBLE — **LEAKY**, label-only

### `slot_features_60m.parquet` (27.3M rows, 17 cols)

- One row per (token_id, block_slot) with trades
- Slots without trades are **not materialised** — this is important: a gap in seconds_since_deploy means zero volume, not missing data
- `holders_count` INT — valid at each slot, grows monotonically in practice
- `top_wallet_bought` BOOL — True if a "smart money" wallet transacted in this slot
- `price_sol_per_token` DOUBLE — some slots may be 0.0 if only sells with no price observation (treat 0 as missing for price features)

### `deployer_actions_60m.parquet` (52.6M rows, 8 cols)

- One row per (token_id, block_slot, seconds_since_deploy, deployer_action)
- `deployer_action` examples: `pump:create_v2`, `pump:buy`, `pump:sell`, `pump:buy_exact_sol_in`, `pump_amm:sell`, `PumpFunSwap:defi_token_swap`, `closeAccount:spl_close_account`, `pump:collect_creator_fee`, `FLASHX8...`: Unknown (likely MEV bot addresses)
- The `FLASHX8...` Unknown action is the most common (14.7M rows) — likely Jito bundle tips or MEV infrastructure; NOT a deployer action
- First deployer sell timing: p10=1s, p25=3s, p50=21s, p90=899s. 55% sell within 30s.

## Survival Statistics

| Threshold | Tokens with trades past t | % of 500k |
|---|---|---|
| t ≥ 0 s | 470,394 | 94.1% |
| t ≥ 60 s | 283,995 | 56.8% |
| t ≥ 300 s (5 min) | 148,070 | 29.6% |
| t ≥ 600 s (10 min) | 112,054 | 22.4% |
| t ≥ 1800 s (30 min) | 48,321 | 9.7% |
| t ≥ 3500 s (~1h) | 8,382 | 1.7% |

Most tokens are fully dead within 5 minutes. The 30-min label horizon is generous —
most tokens have already collapsed before it fires.

## Deployer Distribution

- 91,765 unique deployers for 500k tokens → avg 5.4 tokens/deployer
- Top deployer: **8,465 tokens** (1.7% of all tokens from one wallet)
- Pareto: top 1% of deployers account for ~25-30% of volume
- Deployer concentration follows a power law (log-log linear in scatter plot)

## Time Range

- Min deploy: 2026-04-01 12:30:07 UTC
- Max deploy: 2026-04-20 10:23:30 UTC
- Span: 18.9 days

## Known Data Issues

1. `pytz` module required by DuckDB for TIMESTAMP WITH TIME ZONE operations — install it
2. Polars `group_by` returns tuple keys when iterating — must unwrap: `int(k[0]) if isinstance(k, tuple) else int(k)`
3. LightGBM rejects `object` dtype pandas columns — cast all numeric cols to Float64 before passing
4. `qcut(10, allow_duplicates=True)` required for features like `image_hash_seen_total` where >10% of values are identical (0 or 1)
5. Some tokens have `price_sol_per_token = 0` in a slot — these are sell-only slots (no buying happened); filter with `WHERE price_sol_per_token > 0` for label computation
