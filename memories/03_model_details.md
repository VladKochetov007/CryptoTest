# Model Training Details

## Setup

- **Label**: `hit_2x_30m` = `pmax(price, 0..1800s) / first_slot_price ≥ 2.0`
- **Base rate**: 16.0% (range 12.4%–19.0% across time chunks)
- **Training rows**: 470,256 (tokens with first-slot price observation)
- **Validation**: 5-fold expanding-window walk-forward, sorted by `deploy_time_unix`
- **Embargo**: none (label is anchored at slot 0, no forward contamination in time-split itself; deployer-history features are the only temporal concern, handled in SQL)

## Feature Sets

### `instant` (47 features, Part 1 ≤300 ms)
All derived from `tokens` table + ccxt SOL/BTC. See `eda/build_features.py`.

### `with60s` (58 features, Part 1B stage-2)
`instant` + 11 post-launch aggregates (buy/sell vol, holders, top_wallet, trades at 30s and 60s).  
`price_max_60s` / `price_min_60s` were tried and **removed** — leaky (see memories/01).

## Hyperparameters

### LightGBM
```python
objective="binary", metric="auc"
learning_rate=0.05
num_leaves=63
feature_fraction=0.85
bagging_fraction=0.85, bagging_freq=5
min_data_in_leaf=200
scale_pos_weight=(1 - base) / base   # per-fold, compensates imbalance
early_stopping=50
max_rounds=600
```
Categorical: `deployer_wallet_source_cex_name` — native LightGBM categorical.

### XGBoost
```python
objective="binary:logistic", eval_metric="auc"
eta=0.05, max_depth=6
subsample=0.85, colsample_bytree=0.85
min_child_weight=5
scale_pos_weight=(1 - base) / base
tree_method="hist", nthread=-1
early_stopping_rounds=50, num_boost_round=600
```
Categorical: ordinal-encoded (`.cast(pl.Categorical).to_physical()`).

### CatBoost
```python
iterations=600, learning_rate=0.05, depth=6
eval_metric="AUC", random_seed=42
scale_pos_weight=(1 - base) / base
early_stopping_rounds=50
thread_count=-1
```
Categorical: native Pool with cat_features list.

## OOF AUC by Fold

### instant feature set

| Fold | LGBM | XGB | CatBoost | n_train | n_val | base_val |
|---|---|---|---|---|---|---|
| 0 | 0.7954 | 0.7915 | 0.7929 | 78,376 | 78,376 | 13.3% |
| 1 | 0.7066 | 0.7005 | 0.7003 | 156,752 | 78,376 | 17.6% |
| 2 | 0.7476 | 0.7458 | 0.7452 | 235,128 | 78,376 | 17.3% |
| 3 | 0.7841 | 0.7807 | 0.7845 | 313,504 | 78,376 | 17.3% |
| 4 | 0.7842 | 0.7823 | 0.7824 | 391,880 | 78,376 | 17.5% |
| **OOF** | **0.7653** | **0.7630** | **0.7640** | — | — | — |

Fold 1 anomaly (AUC 0.70): train base rate 13.1% → test base rate 17.6%. The model
trained on a low-base regime mis-calibrates on a higher-base regime. This is the
regime-drift problem in miniature.

### with60s feature set (leaky cols removed)

| Fold | LGBM | XGB | CatBoost |
|---|---|---|---|
| 0 | 0.9635 | 0.9637 | 0.9630 |
| 1 | 0.9171 | 0.9174 | 0.9157 |
| 2 | 0.9351 | 0.9344 | 0.9325 |
| 3 | 0.9544 | 0.9541 | 0.9525 |
| 4 | 0.9519 | 0.9512 | 0.9508 |
| **OOF** | **0.9452** | **0.9449** | **0.9436** |

## Calibration

- Method: `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on OOF predictions
- Input: LightGBM OOF probabilities (best model)
- Score = `calibrated_prob × 100`, clipped to [0, 100]

### Threshold Table (calibrated)

| Threshold | Selection rate | Precision | Lift | Notes |
|---|---|---|---|---|
| 0.39 | 5.2% | 58.3% | 3.51× | **Soft buy** (probe tier) |
| 0.50 | ~3.5% | ~64% | ~4× | Conservative soft |
| 0.70 | ~1.5% | ~75% | ~4.5× | — |
| 0.95 | 0.12% | 99.4% | 5.99× | **High-conviction** (full-size tier) |
| 0.99 | 0.06% | 100% | 6.02× | Maximum precision, minimal selection |

## Deployer-Grouped CV (robustness)

| Fold | AUC | PR-AUC | Val deployers | Val base |
|---|---|---|---|---|
| 0 | 0.7819 | 0.4391 | 17,334 | 14.9% |
| 1 | 0.7760 | 0.4270 | 17,339 | 14.7% |
| 2 | 0.7807 | 0.4455 | 17,339 | 16.6% |
| 3 | 0.7809 | 0.4603 | 17,339 | 17.0% |
| 4 | 0.7727 | 0.4511 | 17,339 | 16.6% |
| **mean** | **0.7784** | **0.4446** | — | — |

Higher than time-fold (0.765) — confirming alpha is not deployer-memorisation.
