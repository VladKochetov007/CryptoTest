# Project: Pump.fun Pre-Buy Scoring & Exit Strategy

**Updated: 2026-04-28 — meta-features added, AUC 0.8014**

## Summary

End-to-end quant prototype on 500k Pump.fun tokens (2026-04-01 → 2026-04-20, 19 days).

## Deliverables

| File | Purpose |
|---|---|
| `eda/build_features.py` | Feature pipeline (47 deploy-time + 11 post-60s + macro) |
| `eda/train.py` | Walk-forward LGBM/XGB/CatBoost training |
| `eda/explain.py` | SHAP + isotonic calibration + 0-100 score + 1k CSV |
| `eda/backtest.py` | 6 exit strategies × 3 universes |
| `eda/robustness.py` | Deployer-grouped CV robustness check |
| `eda/two_stage_sim.py` | Integrated probe→scale→exit simulation with slippage |
| `eda/report.md` | Full narrative report |
| `eda/framing.md` | Senior-quant framing of label, validation, features |

## Key Numbers

| Metric | Baseline | With Meta |
|---|---|---|
| Instant OOF AUC | 0.7653 | **0.8014** (+3.6 pts) |
| With-60s OOF AUC | 0.9452 | — |
| Deployer-grouped CV AUC | 0.778 | — |
| Soft buy threshold | score≥39 → 3.5× lift | — |
| High-conviction threshold | score≥95 → 6× lift | — |
| Best exit (trailing-30) | win 61%, median ROI +10% | — |
| Two-stage PnL | +3352 SOL, 5000 trades | — |

**Production model**: `meta__lgbm` (AUC 0.8014). Top feature: `deployer_hr_7d` (rolling 7d hit rate, IV=0.97).

## Random Seeds Used

| Usage | Seed |
|---|---|
| `np.random.default_rng(42)` | Backtest universe subsampling (model_top, cex caps) |
| `np.random.default_rng(42)` | Two-stage sim subsampling |
| CatBoost `random_seed=42` | Model training |
| LightGBM / XGB | No explicit seed (libraries default to 0) |

## Data Files (not in git)

| File | Size | Description |
|---|---|---|
| `tokens.parquet` | 114 MB | 500k tokens, metadata + deployer fields |
| `slot_features_60m.parquet` | 1.3 GB | 27M rows, per-slot trade aggregates |
| `deployer_actions_60m.parquet` | 149 MB | 52M rows, on-chain deployer actions |
| `eda/features.parquet` | 39 MB | Base feature table (500k × 78 cols) |
| `eda/meta_features.parquet` | ~20 MB | Meta-features (500k × 35 cols) |

## New Scripts Added (2026-04-28)

| Script | Purpose |
|---|---|
| `eda/build_meta_features.py` | Multi-scale deployer/funder/handle rolling + text features |
| `eda/meta_train.py` | LGBM on 77 features (base + meta), SHAP |
| `eda/meta_eda_plots.py` | IV bar, AUC comparison, diagnostic plots |
| `eda/cross_target.py` | 4×4 AUC/lift matrix across label thresholds |
| `eda/stack_distill.py` | Stack-blend OOF + distill (conclusion: don't blend) |

## Cross-Target Key Finding
Training on hit_5x does NOT improve hit_2x detection (hypothesis refuted).
Best lift@10% for hit_2x: train on hit_2x (2.875× vs 2.608× for hit_5x).
For high-conviction tier (5x+): train on hit_5x specifically (lift 3.885× vs 3.757×).
