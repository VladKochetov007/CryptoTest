# Project: Pump.fun Pre-Buy Scoring & Exit Strategy

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

- **Instant AUC** 0.765 (5-fold WF OOF) / 0.778 (deployer-grouped)
- **With-60s AUC** 0.945 (post-launch lookback, no leaky price cols)
- **Soft buy threshold** score ≥ 39 → 3.5× lift, 58% precision, 5% selection
- **High-conviction** score ≥ 95 → 6× lift, 99.4% precision, 0.12% selection
- **Best exit strategy** trailing-30 on model top-decile: median ROI +10%, win 61%
- **Two-stage PnL** +3352 SOL gross on 5000 trades (tail-harvester)

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
| `eda/features.parquet` | 39 MB | Engineered feature table (500k × 70 cols) |
