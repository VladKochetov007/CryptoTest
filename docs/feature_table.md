# Feature Reference — meta__lgbm (85 features, AUC 0.8007)

All features are available at deploy time (≤300 ms after `pump:create_v2` event) unless
noted as **post-60s** (only used by the `with60s` stage-2 model, not the pre-buy scorer).
SHAP rank is by mean |SHAP| on the OOF set.

## Deployer wallet economics (6)

| Feature | SHAP | Description |
|---|---|---|
| `deployer_deposit_amount` | 6 | SOL committed to the bonding curve at creation |
| `deployer_wallet_balance_before` | 11 | Deployer SOL balance before the create tx |
| `deployer_wallet_balance_after_sol` | 3 | Deployer SOL balance after the create tx |
| `deployer_wallet_source_amount_sol` | 20 | SOL received from the funder wallet that seeded this deployer |
| `is_cex` | 83 | 1 if the funder wallet is a labelled CEX withdrawal address |
| `deployer_wallet_source_cex_name` | 74 | Name of CEX (categorical; `__missing__` when not CEX) |

## Token metadata booleans (5)

| Feature | SHAP | Description |
|---|---|---|
| `has_image` | 27 | Token uploaded an image |
| `has_desc` | 73 | Token has a description string |
| `has_website` | 34 | Token has a website URL |
| `has_twitter` | 53 | Token has a Twitter/X handle |
| `has_telegram` | 45 | Token has a Telegram link |

## Token metadata lengths (3)

| Feature | SHAP | Description |
|---|---|---|
| `name_len` | 40 | Character count of token name |
| `ticker_len` | 26 | Character count of ticker symbol |
| `desc_len` | 19 | Character count of description text |

## Deployer all-time history (4)

Computed via cumulative sums with a strict `t < deploy_time` window so each token
sees only strictly earlier deployments from the same wallet.

| Feature | SHAP | Description |
|---|---|---|
| `deployer_prior_n` | 18 | Lifetime count of prior tokens by this wallet |
| `deployer_prior_grad` | 69 | Count of prior tokens that graduated (~$69k market cap) |
| `deployer_prior_hit20k` | 14 | Count of prior tokens that hit $20k market cap (coarse hit_2x proxy) |
| `deployer_seconds_since_last` | 2 | Seconds since this deployer's previous token deploy |

## Funder all-time history (3)

Same strict-past window as deployer history, applied to the source/funder wallet.

| Feature | SHAP | Description |
|---|---|---|
| `funder_prior_n` | 21 | Lifetime count of tokens funded by this source wallet |
| `funder_prior_hit20k` | 41 | Count of prior tokens from this funder that hit $20k |
| `funder_prior_grad` | 60 | Count of prior tokens from this funder that graduated |

## Market activity (3)

Cross-section signals lagged by 1 minute to avoid within-bucket lookahead.

| Feature | SHAP | Description |
|---|---|---|
| `deploys_prev_15m` | 48 | Total new token deploys in the previous 15 minutes |
| `deploys_prev_60m` | 44 | Total new token deploys in the previous 60 minutes |
| `hit20k_rate_prev_60m` | 49 | Rolling hit_2x rate for all tokens in the previous 60 minutes (shifted 1 min) |

## Clone / reuse signals (3)

| Feature | SHAP | Description |
|---|---|---|
| `image_hash_seen_total` | 5 | How many prior tokens share the same image hash (copy-paste content farm signal) |
| `same_ticker_today_prev` | 15 | Count of tokens with the identical ticker deployed today, strictly before this one |
| `same_name_prev_hour` | 32 | Count of tokens with the identical name in the past hour, strictly before this one |

## Mint / deployer address signals (2)

`mint_suffix_PUMP` was present in training but had zero variance and was dropped by LightGBM.

| Feature | SHAP | Description |
|---|---|---|
| `mint_suffix_pump` | 10 | 1 if the mint address ends in "pump" (lowercase vanity address) |
| `deployer_suffix_pump` | 84 | 1 if the deployer address ends in "pump" |

## Name / ticker text (5)

| Feature | SHAP | Description |
|---|---|---|
| `name_alpha_chars` | 43 | Count of alphabetic characters in token name |
| `name_upper_chars` | 7 | Count of uppercase letters in token name |
| `name_word_count` | 16 | Word count of token name (split on whitespace/punctuation) |
| `name_digit_count` | 80 | Count of digit characters in token name |
| `name_has_meme_kw` | 51 | 1 if name contains common meme keywords (DOGE, PEPE, TRUMP, etc.) |

## Ticker text (5)

| Feature | SHAP | Description |
|---|---|---|
| `ticker_len` | 26 | Character count of ticker symbol (also in metadata lengths section) |
| `ticker_digit_count` | 81 | Count of digits in ticker |
| `ticker_special_count` | 39 | Count of non-alphanumeric characters in ticker |
| `ticker_len_4_5` | 76 | 1 if ticker length is exactly 4 or 5 characters |
| `ticker_has_meme_kw` | 58 | 1 if ticker contains meme keywords |

## Clock features (7)

Cyclical encoding prevents discontinuity at midnight/week boundary.

| Feature | SHAP | Description |
|---|---|---|
| `utc_sin` | 35 | sin(2π × utc_hour / 24) — cyclical UTC hour encoding |
| `utc_cos` | 42 | cos(2π × utc_hour / 24) — cyclical UTC hour encoding |
| `utc_hour` | 52 | UTC hour of day (0–23), linear |
| `utc_dow` | 66 | Day of week (0 = Monday, 6 = Sunday) |
| `ny_hour` | 59 | New York local hour (UTC-5 / UTC-4 DST) |
| `ldn_hour` | 68 | London local hour |
| `tokyo_hour` | 67 | Tokyo local hour |

## Macro market — SOL/BTC Binance OHLCV (8)

5-minute bar that closes before the deploy timestamp (no lookahead).

| Feature | SHAP | Description |
|---|---|---|
| `sol_close` | 38 | SOL/USDT close price |
| `sol_vol_1h` | 61 | SOL rolling 1-hour trade volume |
| `sol_vol_24h` | 54 | SOL rolling 24-hour trade volume |
| `sol_ret_1h` | 62 | SOL 1-hour log return |
| `sol_ret_24h` | 64 | SOL 24-hour log return |
| `btc_close` | 17 | BTC/USDT close price |
| `btc_vol_1h` | 63 | BTC rolling 1-hour trade volume |
| `btc_ret_1h` | 55 | BTC 1-hour log return |

---

## META FEATURES (38 new columns added in Phase B)

### Deployer multi-scale rolling — `deployer_prior_n_{window}` and `deployer_hr_{window}`

Rolling count and hit_2x rate for this deployer over the past `{window}`.
Computed with tie-safe `searchsorted(side="left")` so same-second peers never
see each other's outcomes. All windows are strictly `deploy_time < current_deploy_time`.

| Feature | SHAP | Description |
|---|---|---|
| `deployer_prior_n_1h` | 46 | Deployer token count in past 1 hour |
| `deployer_prior_n_6h` | 57 | Deployer token count in past 6 hours |
| `deployer_prior_n_24h` | 31 | Deployer token count in past 24 hours |
| `deployer_prior_n_7d` | 13 | Deployer token count in past 7 days |
| `deployer_hr_1h` | 12 | Deployer hit_2x rate in past 1 hour (hits / count, 0 when count=0) |
| `deployer_hr_24h` | 4 | Deployer hit_2x rate in past 24 hours |
| `deployer_hr_7d` | **1** | Deployer hit_2x rate in past 7 days — **#1 SHAP feature** |

### Funder multi-scale rolling — `funder_prior_n_{window}` and `funder_hr_{window}`

Same rolling logic applied to the source/funder wallet instead of the deployer.

| Feature | SHAP | Description |
|---|---|---|
| `funder_prior_n_24h` | 33 | Funder-seeded token count in past 24 hours |
| `funder_hr_24h` | 22 | Funder hit_2x rate in past 24 hours |
| `funder_prior_n_7d` | 50 | Funder-seeded token count in past 7 days |
| `funder_hr_7d` | 9 | Funder hit_2x rate in past 7 days |

### Funder graph features (5)

Lifetime aggregates over the funder's full past deployer population.

| Feature | SHAP | Description |
|---|---|---|
| `funder_seconds_since_last` | 25 | Seconds since this funder's previous fund operation |
| `funder_unique_deployers_prior` | 70 | Count of distinct deployer wallets ever seeded by this funder (strictly prior) |
| `funder_concentration_hhi` | 28 | HHI of this funder's deployer distribution — high HHI = one deployer dominates |
| `funder_avg_deposit_sol_prior` | 36 | Rolling mean of SOL amounts funded per deployment (strictly prior) |
| `funder_is_dust_funder` | 78 | 1 if funder sends <0.5 SOL AND has >5 prior deployments (drip-fund mule pattern) |

### Twitter handle sybil (1)

| Feature | SHAP | Description |
|---|---|---|
| `handle_unique_deployers_7d` | 23 | Distinct deployer wallets that used this Twitter handle in past 7 days — high count = shared/bot handle |

### Twitter handle static (6)

| Feature | SHAP | Description |
|---|---|---|
| `handle_len` | 37 | Character length of Twitter handle |
| `handle_digit_ratio` | 56 | Fraction of digits in handle |
| `handle_has_underscore` | 75 | 1 if handle contains an underscore |
| `handle_ends_in_digits` | 79 | 1 if handle ends with digit characters |
| `handle_is_celeb` | 47 | 1 if handle matches a known celebrity/influencer name list |
| `handle_contains_ticker` | 65 | 1 if Twitter handle contains the token ticker |

### Twitter handle rolling (2)

| Feature | SHAP | Description |
|---|---|---|
| `handle_prior_n_24h` | 24 | Count of tokens using this handle in past 24 hours |
| `handle_hr_24h` | 8 | hit_2x rate for tokens using this handle in past 24 hours |

### Description text (6)

| Feature | SHAP | Description |
|---|---|---|
| `desc_has_url` | 71 | 1 if description contains a URL |
| `desc_has_deployed_template` | 72 | 1 if description matches a known copy-paste launch template |
| `desc_exclamation_count` | 82 | Count of exclamation marks in description |
| `desc_template_score` | 77 | Fraction of description matched by template phrase patterns |
| `desc_word_count` | 30 | Word count of description |
| `desc_has_address` | 85 | 1 if description contains a Solana wallet address |

### Meme keyword rolling (1)

| Feature | SHAP | Description |
|---|---|---|
| `meme_kw_hr_24h` | 29 | Rolling hit_2x rate over past 24h for tokens sharing the same meme keyword — near-zero IV; kept for interactions |

---

## Post-60s features (11) — stage-2 model only

These are NOT available at the ≤300 ms pre-buy decision. Used only by the `with60s` model
to scale into or abort a probe position taken at slot 0.

| Feature | Description |
|---|---|
| `buy_vol_30s` | SOL buy volume in first 30 seconds |
| `sell_vol_30s` | SOL sell volume in first 30 seconds |
| `active_slots_30s` | Count of slots with at least one trade in first 30 seconds |
| `holders_30s` | Unique holder count at 30 seconds |
| `top_wallet_30s` | Largest single-wallet token fraction at 30 seconds |
| `trades_30s` | Total trade count in first 30 seconds |
| `buy_vol_60s` | SOL buy volume in first 60 seconds |
| `sell_vol_60s` | SOL sell volume in first 60 seconds |
| `holders_60s` | Unique holder count at 60 seconds |
| `top_wallet_60s` | Largest single-wallet token fraction at 60 seconds |
| `trades_60s` | Total trade count in first 60 seconds |
