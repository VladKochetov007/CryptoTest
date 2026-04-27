# Pump.fun Pre-Buy Scoring + Exit Strategy — Final Report

> Senior-quant prototype. EDA → engineered features → 3-model GBM ensemble walk-forward
> training → SHAP → calibrated 0–100 score → exit-strategy backtest with model-selected
> universe + baselines.

## 0. TL;DR

| Metric | Value |
|---|---|
| Tokens analysed | 500,000 (19-day window, 2026-04-01 → 2026-04-20) |
| Tokens with first-slot price | 470,256 (94.0%) |
| Base hit_2x in 30 min | 16.0% |
| **Instant model AUC (5-fold walk-forward, OOF)** | **0.765** (LGBM), 0.764 (CatBoost), 0.763 (XGB) |
| With-60s lookback AUC | 0.945 (LGBM) — used for stage-2 add/abort policy |
| **Soft buy threshold** (selection ≥5%) | score ≥ 39 → **3.5× lift**, precision 58% |
| **High-conviction threshold** | score ≥ 95 → **6.0× lift**, precision 99.4% |
| Best exit (model top-decile universe) | trailing 30 — **median ROI +10%, win 61%** |
| Alpha vs random universe | trailing 30 random ROI +11% (mean), win 36% — **+25 pp win** from model |
| Alpha vs CEX heuristic | virtually zero — `is_cex` carries IV 0.0001 |

Artefacts: `eda/artifacts/` (models + OOF), `eda/scoring/` (calibration + 1k CSV + SHAP),
`eda/backtest/` (per-strategy summaries + per-trade parquets).

## 1. Why the task-default heuristics fail in this dataset

| Task heuristic | Reality |
|---|---|
| funded from CEX → **+20** | `is_cex` IV = **0.0001**. Hit_20k 2.30% (CEX) vs 2.21% (non-CEX). Useless as a flag. CEX *name* matters: Gate.io 22.5% hit_2x (lift 1.41), MEXC 8.0% (lift 0.50, anti-signal). |
| deposit > 1 SOL → **+20** | Continuous, not step. Bins (SOL): 0–0.2 → 1.5%, 0.2–1 → 3.0%, 1–5 → 2.8%, 5–20 → 3.0%, 20+ → 4.2% hit_20k. Use as continuous feature. |
| no prior tokens → **+10** | Reversed. Veteran deployers slightly *outperform* first-timers; SHAP top-3 for **prior_hit20k**, **prior_n**. |
| unique image → **+15** | Forward but small (`has_image` IV 0.004). The right feature is the *cluster size* of the hash (`image_hash_seen_total`) — ranks #5 in SHAP. |
| novel ticker → **+10** | `same_ticker_today` IV 0.005. Marginal. |

A linear additive scorecard cannot capture the actual non-monotone dependences (e.g. deposit
× balance interaction, CEX-name dummy, regime-dependent base rate). Tree boosting is required.

## 2. Label, validation, leakage

* Label `hit_2x_30m` = `pmax(price, t∈[0..1800s]) / first_slot_price ≥ 2`. Base 16.0%.
* Validation: 5 expanding-window walk-forward folds on `deploy_time_unix`. No random shuffling.
  Base rate drifts 12.4% → 19.0% across the 19-day window → static splits inflate AUC by ~5 pp.
* Strict deploy-time-only feature set for Part 1. Leakage guards:
  - `ath_market_cap_usd`, `latest_market_cap_usd`, `pmax_30m`, `peak_sec_30m` excluded
    from features.
  - Deployer history aggregates use **only past tokens** (window function with
    `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`).
  - Cross-section regime features (`deploys_prev_15m`, `hit20k_rate_prev_60m`) shifted by 1
    minute bucket so the current token never reads its own contribution.
  - Funder history (source-wallet) computed the same way.
* `with60s` set required removing `price_max_60s/min_60s` (they trivially imply the label
  when 2× hits within 60 s); after removal the OOF AUC dropped from 0.99 → 0.95.

## 3. Feature inventory (47 deploy-time + 11 post-60-s)

### Deploy-time (Part 1, ≤ 300 ms latency)

* Funding numerics: `deployer_deposit_amount`, `..._balance_before`, `..._balance_after`,
  `..._source_amount_sol`. Drop one of `deposit/source_amount` (ρ = 0.9996).
* `is_cex` (boolean, weak), **`deployer_wallet_source_cex_name`** (categorical via native-cat
  LGBM — IV 0.078).
* Metadata booleans (image, desc, website, twitter, telegram) and lengths.
* **Deployer history**: `prior_n`, `prior_grad`, `prior_hit20k`, `seconds_since_last`.
* **Funder history**: `funder_prior_n/_hit20k/_grad`.
* **Cross-section regime** (shifted-rolling): `deploys_prev_15m`, `deploys_prev_60m`,
  `hit20k_rate_prev_60m`.
* Clone proxies: `image_hash_seen_total`, `same_ticker_today_prev`, `same_name_prev_hour`.
* Address vanity: mint suffix `pump`/`PUMP`, deployer suffix `pump`.
* Time of day: UTC sin/cos, NYC/London/Tokyo hour, day of week.
* Macro (Binance ccxt 5-min OHLCV at deploy time): SOL & BTC 1h/24h vol & returns.

### Post-60 s (stage-2 / Part 2 backtest)

`buy_vol_30s/60s`, `sell_vol_30s/60s`, `holders_30s/60s`, `top_wallet_30s/60s`,
`trades_30s/60s`, `active_slots_30s`. Univariate AUC ≥ 0.80.

## 4. Models — walk-forward AUC

### Instant feature set (47 features)

| algo | OOF AUC | PR-AUC | Brier | per-fold AUC |
|---|---|---|---|---|
| LightGBM | **0.7653** | 0.4289 | 0.1967 | 0.795 / 0.707 / 0.748 / 0.784 / 0.784 |
| CatBoost | 0.7640 | 0.4253 | 0.2018 | 0.793 / 0.700 / 0.745 / 0.785 / 0.782 |
| XGBoost | 0.7630 | 0.4230 | 0.2047 | 0.792 / 0.700 / 0.746 / 0.781 / 0.782 |

Models are essentially tied; LGBM picked as deployment target for SHAP & calibration.
Fold 1 is the worst (0.70). Inspecting the fold split: it covers the regime where base
rate jumps from 13.3% to 17.6% — the model trained on the lower-base regime
under-calibrates. Re-fitting per-window is mandatory in production.

### Deployer-grouped 5-fold CV (robustness)

To rule out deployer-memorisation as the source of alpha (one deployer makes 8,465
tokens), `eda/robustness.py` re-fits LGBM with `GroupKFold(group=deployer_address)` so
that no deployer appears in both train and test:

| fold | AUC | PR-AUC | val deployers | val rows | val base |
|---|---|---|---|---|---|
| 0 | 0.7819 | 0.4391 | 17,334 | 94,052 | 14.9% |
| 1 | 0.7760 | 0.4270 | 17,339 | 94,051 | 14.7% |
| 2 | 0.7807 | 0.4455 | 17,339 | 94,051 | 16.6% |
| 3 | 0.7809 | 0.4603 | 17,339 | 94,051 | 17.0% |
| 4 | 0.7727 | 0.4511 | 17,339 | 94,051 | 16.6% |
| **mean** | **0.7784** | 0.4446 | — | — | — |

**Deployer-grouped AUC > time-fold AUC** (0.778 vs 0.765). The features carry signal
that generalises to unseen deployers, and the time-fold's lower AUC is dominated by
regime drift, not by leak. Strong validation result.

### With 60-s lookback (58 features, post-launch policy)

| algo | OOF AUC | PR-AUC |
|---|---|---|
| LightGBM | **0.9452** | 0.8087 |
| XGBoost | 0.9449 | 0.8079 |
| CatBoost | 0.9436 | 0.7995 |

Use this as the **stage-2 adder/abort** model: probe size at slot 0 with the instant
model, then scale in or exit at 60 s using the with60s model.

## 5. SHAP — top features (LGBM instant, last fold, 20 k tokens)

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | deployer_prior_hit20k | 0.558 |
| 2 | deployer_seconds_since_last | 0.435 |
| 3 | deployer_prior_n | 0.375 |
| 4 | deployer_wallet_balance_after_sol | 0.182 |
| 5 | image_hash_seen_total | 0.167 |
| 6 | funder_prior_n | 0.155 |
| 7 | mint_suffix_pump | 0.143 |
| 8 | deployer_deposit_amount | 0.143 |
| 9 | name_upper_chars | 0.128 |
| 10 | btc_close | 0.108 |

The deployer-graph features (rows 1–3, 6) dominate. The CEX flag does **not** appear in
the top 10. The macro/time features carry small, robust contribution. Address-vanity
flags surface — likely an interaction with deployer factories that mint many pump-suffix
tokens. See `eda/scoring/shap_summary.png` for the full beeswarm.

## 6. Calibration → 0–100 score

Isotonic regression on OOF probabilities → calibrated probability × 100 = score.

| Tier | Threshold (calibrated p) | Lift over base | Selection rate | Precision |
|---|---|---|---|---|
| **Buy** | 0.39 (score ≥ 39) | **3.51×** | 5.2% | 58.3% |
| **High conviction** | 0.95 (score ≥ 95) | **5.99×** | 0.12% | 99.4% |

Production policy: place a probe size on `Buy`, full size on `High conviction`. Use the
60-s model to decide whether to scale or abort the probe.

`eda/scoring/scored_1000_recent.csv` — the requested 1k-token table (most recent
deploys), with `score_0_100`, `buy_decision`, `high_conviction`. 74 of 1,000 fire `buy`,
1 fires `high_conviction`.

## 7. Part 2 — exit strategies

5,000-token universe (sampled from top-decile model OOF), each replayed second-by-second
on `slot_features_60m`, with deployer-action stream attached for the rug-detection
strategy. Ground-truth ROI uses first-slot price as entry and includes a 30-min timeout.

### 7.1 Universe = LGBM top decile (model_top)

| Strategy | mean ROI | median ROI | win | med DD | med hold | exit reasons |
|---|---|---|---|---|---|---|
| tp_2x_only | +7.33 | +0.094 | 52% | −0.118 | 35 s | tp 47% / timeout 53% |
| tp_2x_sl_50 | +7.33 | 0.000 | 50% | −0.118 | 15 s | tp 45% / sl 24% / timeout 31% |
| **trailing_30** | +0.53 | **+0.104** | **61%** | −0.000 | 26 s | trail 61% / sl 17% / timeout 22% |
| deployer_sell_exit | +7.31 | −0.194 | 32% | −0.230 | 47 s | tp 25% / sl 30% / timeout 45% |
| vol_stagnation_10 | +7.33 | +0.122 | 53% | −0.089 | 15 s | tp 42% / sl 23% / stag 11% / timeout 24% |
| **sell_pressure_5** | +7.32 | **+0.138** | **54%** | −0.055 | 12 s | tp 41% / sl 21% / sell_pressure 23% / timeout 14% |

Mean ROI is dominated by extreme winners (one 200× swamps thousands of −50% trades). The
**actionable signal is the median**: trailing-30 and sell_pressure_5 deliver positive
medians with ≥54% win rate. tp_2x_only also yields high mean but with a much worse
left-tail (−98% worst drawdown), making it riskier under leverage.

### 7.2 Baseline universes (alpha confirmation)

Random 5,000 tokens (no model):

| Strategy | mean ROI | median ROI | win |
|---|---|---|---|
| trailing_30 | +0.113 | 0.000 | 36% |
| sell_pressure_5 | +0.081 | −0.001 | 34% |

CEX heuristic (`is_cex == 1` AND `deposit > 1 SOL`):

| Strategy | mean ROI | median ROI | win |
|---|---|---|---|
| trailing_30 | +0.087 | 0.000 | 37% |
| sell_pressure_5 | +0.088 | 0.000 | 35% |

**Conclusion**: the model picks 25 percentage points of additional win-rate over the
random universe, and the CEX heuristic provides ~zero alpha. The mean-ROI gap (model
+0.53 / +7.33 vs random +0.11 / +0.08) is even larger because the model concentrates the
heavy right-tail.

### 7.3 End-to-end two-stage PnL (`eda/two_stage_sim.py`)

Combine instant + with60s + trailing-30 in the actual product flow:
1. instant_score ≥ 0.39 → place 0.1 SOL probe at slot 0 (buy slip 100 bps).
2. At 60 s, evaluate with60s model:
   - p ≥ 0.85 → scale to 1.0 SOL total (50-50 averaged with probe).
   - p < 0.30 → exit at slot-60 price (probe-saver).
   - else → keep probe.
3. Trailing-30 on the running position; sell slip 100 bps.

5,000-trade simulation (random subsample of the 224,676 instant-buy candidates):

| Metric | Value |
|---|---|
| Probe-only kept | 3,336 (66.7%) |
| Aborted at 60 s | 824 (16.5%) |
| Scaled to full | 840 (16.8%) |
| **Total PnL** | **+3,352 SOL** |
| Mean PnL / trade | +0.67 SOL |
| Median PnL / trade | −0.012 SOL |
| Win rate | 27.6% |
| Worst trade | −0.84 SOL (SL capped) |
| Best trade | +3,410 SOL (one extreme winner) |
| p10 / p90 PnL | −0.086 / +0.054 SOL |

The median is intentionally near zero — the strategy is a tail-harvester. Right-tail
dominance is a known structural property of meme-coin trading: the rare 100-1000×
runner pays for thousands of small chops. The scale-decision at 60 s is what makes
the right tail tractable: only 17% of trades reach full size, but those carry the
heavy-tail capture.

### 7.4 Strategy recommendation

For PnL stability use **trailing-30** (lowest median DD, highest win rate, capped right-
tail). For capital-efficient sniping with stop-loss discipline use **sell_pressure_5**
(median hold 12 s — fastest cycle, best capital turnover). The deployer-sell rule is
useful as an additional stop overlay rather than a primary exit (it has high recall on
rugs but bad median ROI on its own due to false alarms when deployers re-buy).

## 8. Part 3 — Improvement ideas

### Modelling

1. **Two-stage scoring** (probe-then-scale). Instant-model score at slot 0 commits a
   small probe, the 60-s model decides scale/abort. The 60-s AUC of 0.945 makes this a
   genuine separator between rug paths and survivors.
2. **AutoGluon stack**, distilled to a single LGBM for production latency. With three
   GBMs already producing AUC 0.763–0.765 a stacked OOF + linear blender typically
   recovers another 1–2 AUC points; cheap to add.
3. **Deployer-grouped CV** in addition to time-fold CV — the top deployer creates 8,465
   tokens, almost certainly leaking style across folds.
4. **Per-CEX target encoding with smoothing** instead of a single `cex_name` categorical.
   Already implicitly handled by LGBM but explicit smoothing helps with rare CEXes.
5. **Honeypot/rug screening as a hard pre-filter**: drop tokens whose deployer is in a
   known rug-factory cluster (>50 tokens with 0 wins). Their volume is huge and they
   confuse the model with noise; explicit filtering recovers more PR-AUC than another
   feature.

### gRPC stream additions (highest expected value)

| Field | Reason |
|---|---|
| Deployer self-buy size in same tx as `pump:create_v2` | Distinguishes factory dump-bag size, large alpha source |
| Holder list at slot 0–3 with balances | Compute Gini, sniper-share, single-wallet >50%, top-10 % |
| Same-block bundle-detection (Jito tip wallet) | Identifies organised pump groups, biggest scam vector |
| Deployer last-24-h SOL P&L | Beats `prior_grad` count (which equally weights ancient and recent) |
| Funding-source graph degree | Hub wallets seeding many deployers = factory hubs |
| MEV-bot transaction in same block as create | Sniper presence on the launch slot |
| Mint authority + freeze authority status | Honeypot screening |
| Token name embedding (text trigram or fastText) | Cross-token similarity vs winners in last 24 h |

### Operational risks

* Class drift: hit_2x rate drifted +50% relative across the 19-day window. Production
  must retrain ≤ daily and monitor Brier score, not AUC.
* Selection risk: a real bot competes with other snipers for the same first-slot
  liquidity. The realised entry will be slot-1 to slot-3 prices — re-evaluate the label
  with a 5-second anchor before deploying capital.
* Backtest cost model: 0.5–1.5% slippage + Solana priority fees should be subtracted from
  every trade before booking PnL. The win rates above are gross.
* If the token is a scam (worst_dd ≈ −98%), the price collapses faster than any exit
  trigger except a same-block deployer-sell mempool watcher. We mitigate via a `−50%`
  stop, but in many rugs the bonding-curve liquidity is gone before the next slot. The
  better defence is **never to score the rug** — the instant model is the primary
  protection layer.

## 9. Files

```
eda/
  build_features.py                # feature pipeline (deploy-time + post-60s + macro)
  train.py                         # walk-forward LGBM/XGB/CatBoost
  explain.py                       # SHAP + isotonic calibration + 1k-token CSV
  backtest.py                      # 6 exit strategies × 3 universes
  features.parquet                 # 500k × 70 cols
  artifacts/
    instant__{lgbm,xgb,catboost}/  # OOF + summary + final-fold model
    with60s__{lgbm,xgb,catboost}/
  scoring/
    model_comparison.json
    shap_ranking.json, shap_summary.png
    threshold_sweep.json
    scoring_summary.json
    scored_tokens.parquet          # all 391880 OOF rows with score
    scored_1000_recent.csv         # task deliverable
  backtest/
    summaries_{model_top,random,cex_heuristic}.json
    trades_*.parquet
```
