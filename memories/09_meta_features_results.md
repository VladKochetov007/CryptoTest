---
name: meta_features_results
description: Meta-feature engineering results — multi-scale deployer windows + text features. +3.6 AUC points.
type: project
---

# Meta-Feature Engineering Results

**Date:** 2026-04-28. Scripts: `eda/build_meta_features.py`, `eda/meta_train.py`, `eda/meta_eda_plots.py`.

## Key Result

**OOF AUC: 0.8014** vs baseline instant/lgbm: **0.7653** → **+3.6 AUC points**

## New Features Added (35 meta-feature columns)

### Multi-Scale Deployer Features (Phase 1)
Computed via cumsum+searchsorted per group — O(N log N), ~7s for 500k tokens.
All windows strict past (deploy_time_unix < current token's time).

| Feature | IV | SHAP rank |
|---|---|---|
| `deployer_hr_7d` | 0.973 | **#1** (0.735) |
| `deployer_hr_24h` | 0.896 | **#4** (0.165) |
| `deployer_hr_6h` | 0.773 | — |
| `deployer_hr_1h` | 0.661 | **#13** (0.067) |
| `deployer_prior_n_7d` | 0.069 | **#9** (0.073) |
| `deployer_prior_n_24h` | 0.067 | — |
| `deployer_prior_n_6h`, `deployer_prior_n_1h` | low | — |

**Critical finding:** `deployer_hr_7d` (7-day rolling hit_2x rate for deployer) has IV=0.97 and is the #1 feature by SHAP. This is far superior to the existing `deployer_prior_hit20k` (all-time cumulative count, SHAP #1 in baseline). The rolling rate captures regime changes better than the all-time count.

### Multi-Scale Funder Features
| Feature | IV |
|---|---|
| `funder_hr_7d` | 0.673 |
| `funder_hr_24h` | 0.622 |
| `funder_prior_n_7d`, `funder_prior_n_24h` | low |

### Twitter Handle Rolling Features
| Feature | IV | SHAP rank |
|---|---|---|
| `handle_hr_24h` | 0.255 | **#8** (0.109) |
| `handle_prior_n_24h` | low | — |

Handles associated with high 24h win rates are a significant positive signal. Deployers reusing the same twitter handle with a track record.

### Static Text Features

| Feature | IV | Notes |
|---|---|---|
| `name_word_count` | 0.058 | 2-word names have higher hit rate |
| `desc_word_count` | 0.042 | moderate |
| `desc_template_score` | 0.031 | more template phrases = lower quality |
| `desc_has_url` | 0.021 | URLs in desc = slightly negative |
| `ticker_digit_count` | ~0 | noise |
| `name_has_meme_kw` | ~0 | not predictive in isolation |
| `ticker_has_meme_kw` | ~0 | noise |
| `handle_len` | low | |
| `handle_is_celeb` | low | too rare |
| `meme_kw_hr_24h` | low | meme keyword rolling rate = no signal |

### Key Negative Finding: Meme Keywords Don't Predict
`name_has_meme_kw` and `meme_kw_hr_24h` have essentially zero IV. Whether a token name contains DOGE/PEPE/TRUMP etc. is not predictive of 2x outcomes. The signal is in the deployer's track record, not the token's branding.

## OOF AUC by Fold (meta LGBM)

| Fold | AUC | n_train | n_val | br_train | br_val |
|---|---|---|---|---|---|
| 0 | 0.8331 | 78,376 | 78,376 | 0.129 | 0.133 |
| 1 | 0.7572 | 156,752 | 78,376 | 0.131 | 0.176 |
| 2 | 0.7827 | 235,128 | 78,376 | 0.146 | 0.173 |
| 3 | 0.8147 | 313,504 | 78,376 | 0.153 | 0.173 |
| 4 | 0.8090 | 391,880 | 78,376 | 0.157 | 0.175 |
| **OOF** | **0.8014** | — | — | — | — |

Regime drift is still present (fold 1 dip from base-rate mismatch), but the meta features improve across all folds vs baseline.

## Top-20 SHAP Features (meta LGBM)

| Rank | Feature | Mean |SHAP| | Is New |
|---|---|---|---|
| 1 | `deployer_hr_7d` | 0.7349 | YES |
| 2 | `deployer_seconds_since_last` | 0.3856 | no |
| 3 | `deployer_wallet_balance_after_sol` | 0.1658 | no |
| 4 | `deployer_hr_24h` | 0.1654 | YES |
| 5 | `image_hash_seen_total` | 0.1604 | no |
| 6 | `name_upper_chars` | 0.1326 | no |
| 7 | `deployer_deposit_amount` | 0.1196 | no |
| 8 | `handle_hr_24h` | 0.1085 | YES |
| 9 | `deployer_prior_n_7d` | 0.0727 | YES |
| 10 | `funder_prior_n` | 0.0701 | no |
| 11 | `mint_suffix_pump` | 0.0691 | no |
| 12 | `deployer_wallet_balance_before` | 0.0684 | no |
| 13 | `deployer_hr_1h` | 0.0672 | YES |
| 14 | `btc_close` | 0.0664 | no |
| 15 | `funder_hr_24h` | 0.0636 | YES |
| 16 | `same_ticker_today_prev` | 0.0567 | no |
| 17 | `deployer_prior_hit20k` | 0.0537 | no |
| 18 | `deployer_wallet_source_amount_sol` | 0.0502 | no |
| 19 | `name_word_count` | 0.0478 | YES |
| 20 | `desc_len` | 0.0441 | no |

**7 of top 20 features are new meta-features.**

## Interpretation

The core insight is that `deployer_hr_7d` (rolling 7-day hit rate for the deployer) is the strongest single feature by a large margin. It's a better version of `deployer_prior_hit20k` because:
1. It normalizes for volume (rate vs count)
2. It's time-windowed (recent performance > all-time cumulative)
3. It adapts to regime changes (a deployer active in a bull regime might have high rates that then decay)

This was missed in Phase 1 because `deployer_prior_hit20k` only counts cumulative wins without normalizing by total launches.

## Files
- `eda/meta_features.parquet` — 35 new features, 500k rows
- `eda/artifacts/meta__lgbm/summary.json`
- `eda/artifacts/meta__lgbm/shap_meta.json`
- `eda/plots/meta_shap_importance.png`
- `eda/plots/meta_feature_iv_bar.png`
- `eda/plots/meta_deployer_scale_auc.png`
- `eda/plots/meta_text_auc.png`
- `eda/plots/meta_meme_kw_by_hour.png`
- `eda/plots/meta_handle_len_hitrate.png`
- `eda/plots/meta_desc_template_hitrate.png`
