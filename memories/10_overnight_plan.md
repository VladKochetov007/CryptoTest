---
name: overnight_research_plan
description: Meta-feature research plan — multi-scale features, cross-target experiment, text features, stack/distill. April 28 2026.
type: project
---

# Overnight Meta-Feature Research Plan

**Goal**: +0.01–0.03 AUC over instant baseline (0.7653). Management-facing tables.

**Why:** Current model uses mostly deployer-history and macro features. Under-exploited: multi-scale temporal dynamics, name/ticker text signal, cross-target transfer (train on 5x → predict 2x), and stacking.

**How to apply:** Run phases in order. Gate on AUC delta. Kill switch if delta < threshold to avoid over-engineering.

## Hard Constraints
- Inference: <50 ms feature extraction + <50 ms scoring (well inside 300–400 ms budget)
- Compute: 16 cores, 6 GB GPU, 64 GB RAM, overnight (~10 h)
- No future leakage: every feature audited before adding to model

## Kill Switches
- Phase 1 multi-scale ΔAUC < +0.005 → skip, document
- Phase 2 cross-target lift < baseline × 1.05 → keep base label
- Phase 3 text features all IV < 0.01 → drop entirely
- Phase 6 distilled AUC < 0.770 OR latency > 50 ms → revert to base LGBM

## Phase 0 — Data audit (15 min)
- Inspect all columns in tokens.parquet — is `twitter` URL or boolean? Is `desc` text available?
- Confirm deployer_actions_60m has no buyer wallet data
- Output: `eda/data_audit.md`

## Phase 1 — Multi-scale deployer/funder features (1–2 h)
Script: `eda/build_meta_features.py`

Rolling windows for deployer: 1h, 6h, 24h, 7d
- `deployer_prior_n_{1h,6h,24h,7d}` — DuckDB range join strict `<`
- `deployer_hit_rate_{1h,24h,7d}` — conditional sum/count
- `deployer_velocity_24h_7d` — ratio of 24h to 7d rate (acceleration)
- `funder_n_24h`, `funder_hit_rate_24h` — same on wallet source

Verification: single-fold LGBM ΔAUC over baseline.

## Phase 2 — Cross-target experiment (1 h)
Script: `eda/cross_target.py`

Labels: hit_2x, hit_3x, hit_5x, hit_10x (base rates ~16/7/3/1%)
Train 4 LGBMs (instant features), cross-evaluate all 16 combinations.

Hypothesis: train on hit_5x, use to predict hit_2x → better lift at top decile, lower risk, same profit (per user's prior experience).

Output: 4×4 table of {AUC, lift, win_rate} × {train_target, eval_target}
Save: `memories/07_cross_target_table.md`

## Phase 3 — Text features on name/ticker (1 h)
Script: `eda/text_features.py`

| Feature | Compute |
|---------|---------|
| `name_has_meme_kw` | regex: DOGE, PEPE, TRUMP, AI, BONK, CAT, SHIB, MOON, ELON |
| `name_meme_kw_24h_winrate` | rolling hit_2x rate for tokens with same keyword, shifted |
| `ticker_char_diversity` | unique_chars / len |
| `ticker_digit_ratio`, `ticker_special_ratio` | factory signal |
| `name_repeated_chars` | max run-length |
| `ticker_recent_collision_24h` | same-ticker count last 24h, strict < time |
| `ticker_tfidf_cosine_winners_24h` | char 3-gram TF-IDF vs last 24h winners |

All temporal cuts use strict `<`. Compute IV/AUC, keep top-10.

## Phase 4 — Slot 0–3 microstructure features (1 h)
Script: `eda/build_micro_features.py`

For with60s set:
- `holders_growth_slope_5s` — linear fit on holders_count first 5 slots
- `buy_sell_imbalance_3s/10s/30s/60s` — vol ratio at multiple lags
- `top_wallet_first_3s` — flag from slot_features
- `vol_zscore_vs_market_60s` — normalize against minute-aggregate market vol

## Phase 5 — ArXiv scan (parallel, 30 min)
Run while Phase 1 trains. Extract feature ideas not yet in our set.
Save: `memories/08_literature_features.md`

NOT doing: fine-tune small LM (names too short, latency budget), custom NN (GBM already sweet-spot for tabular).

## Phase 6 — Stack and distill (1.5 h)
Script: `eda/stack_distill.py`

1. Train 3 GBMs with expanded feature set per fold
2. Blend OOF via isotonic-weighted geometric mean
3. Distill: train small LGBM (num_leaves=31, n_estimators=300) on raw features → blended soft labels
4. Gate: distilled AUC ≥ 0.770 AND latency <50 ms

Production = distilled single LGBM.

## Phase 7 — Leak audit v2 (30 min)
Script: `eda/leak_audit_v2.py`

Three tests per feature:
1. Permutation test (shuffle label, retrain, check feature rank drops to noise)
2. Future-only mask (drop feature, AUC should drop by ~SHAP amount)
3. Pearson(feature, label) per time chunk (leak amplifies over time)

## Phase 8 — Universe/sizing experiments (45 min)
Script: `eda/sim_comparison.py`

Compare baseline vs meta+multi-scale vs meta+cross-target vs meta+Kelly sizing
in two_stage_sim: total PnL, win%, max DD.

## Phase 9 — Report + memo update (30 min)
`report_v2.md` for management:
- AUC progression
- Cross-target 4×4 table
- New top-10 SHAP
- ROI delta on two-stage sim
- 1-page exec summary

## Execution Timeline
```
t=0:00  Phase 0 (data audit)          — serial, gates rest
t=0:15  Phase 1 (multi-scale)         — 8 cores
        Phase 5 (arxiv)               — background
t=1:15  Phase 2 (cross-target)        — 16 cores
t=2:15  Phase 3 (text features)       — 4 cores
        Phase 4 (microstructure)      — 12 cores
t=3:15  Phase 7 (leak audit)          — serial GATE
t=3:45  Phase 6 (stack+distill)       — GPU + 16 cores
t=5:15  Phase 8 (two-stage sim)       — 16 cores
t=6:00  Phase 9 (report+memos)        — serial
t=7:00  Done. 3h buffer for debugging.
```
