# Open Issues & Next Steps

## Known Limitations

### Entry Price Assumption
All backtests use **first-slot mid-price** as entry. A real bot:
1. Competes with MEV snipers for the same block
2. Will typically execute at slot 1–3 (50–400 ms delay)
3. Pays priority fees (Jito tips) on top of base gas

**Fix**: re-label with 5-second-post-deploy price. Expect hit_2x base rate to drop ~2-3 pp.

### Cost Model
Part-2 backtest is gross (no slippage). Two-stage sim uses 100 bps round-trip (realistic
for small Pump.fun trades, but some slots have very thin liquidity → could be 200-500 bps).

### Regime Sensitivity
Hit_2x rate varies 12.4%–19.0% over 19 days (LetsBonk.fun took share from Pump.fun in
this window). Production needs rolling retraining + recalibration ≤ daily.

### OOF Coverage
391,880 of 470,256 label-eligible tokens have OOF predictions (fold-0 train set has
no OOF). The 78,376-token fold-0 gap means the "1k recent tokens" deliverable reads from
a smaller pool. Re-train on all data (no OOF) for production.

### Holder Gini Not Computed
The dataset has no per-holder-address breakdown (only `holders_count`). Gini/HHI/Nakamoto
require the raw holder list (available via gRPC `getTokenLargestAccounts` at slot 0).
This is likely a top-5 feature in a live system — the `deployer_prior_hit20k` proxy is
a reasonable substitute but not the same.

### AutoGluon Not Run
User suggested AutoGluon for stacking. Expected +1–2 AUC on top of single-model 0.765.
Deferred due to time budget. Script skeleton: `eda/autogluon_stub.py` (not written).

## Next Experiments to Run

1. **5-second label anchor**: re-run `build_features.py` with `p0 = price at t=5s` instead
   of `t = first_slot`, then compare OOF AUC. If AUC drops > 0.01, the model is partly
   capturing instant-slot sniper dynamics that won't be replicable.

2. **Deployer-grouped CV + time-fold combined**: use `GroupTimeSeriesSplit` that both
   respects time order and holds out unseen deployer groups. More conservative than either alone.

3. **AutoGluon 30-min run**: install `autogluon.tabular`, fit with `presets="good_quality"`,
   save OOF predictions, compare. Then distill best stack to a single LGBM via soft labels.

4. **Holder Gini at slot 0**: fetch `getTokenLargestAccounts` for a sample of 10k tokens
   from Helius API, compute Gini/HHI, measure AUC uplift, decide whether to add to gRPC
   stream spec.

5. **Sniper-count feature**: count distinct buyer addresses in slots 0–3 from
   `deployer_actions_60m` (where `deployer_action = 'pump:buy'` in the same slot as create).
   High sniper count → coordinated bundle → bearish for retail buyers.

6. **Kelly-optimal sizing**: simulate the two-stage policy with Kelly fraction sizing
   based on calibrated probability, compare sharpe vs fixed-size.

7. **Adversarial deployer filter**: cluster deployers by funding-source graph, identify
   hub wallets seeding >100 deployers with 0 hits. Hard-filter before scoring.

## Technical Debt

| Item | Priority | Effort |
|---|---|---|
| Remove `catboost_info/` from working dir (already gitignored) | low | 1 min |
| Add `requirements.txt` pinned versions | medium | 5 min |
| Parameterise thresholds in `two_stage_sim.py` via CLI args | medium | 30 min |
| Add logging to `train.py` (currently uses print) | low | 20 min |
| `explain.py` re-loads full features to regenerate 1k CSV; slow | low | — |

## Random Seeds — Complete Record

| Location | Seed | Effect |
|---|---|---|
| `backtest.py` L107 `rng = np.random.default_rng(42)` | 42 | Universe subsampling for model_top / random / cex_heuristic |
| `two_stage_sim.py` L142 `rng = np.random.default_rng(42)` | 42 | Universe subsampling for 2-stage |
| CatBoost `random_seed=42` | 42 | Tree node split randomness |
| LightGBM | default=0 (library default) | Feature fraction, bagging |
| XGBoost | library default | same |
| `walkforward_indices` — deterministic `np.linspace` | n/a | No randomness |
| GroupKFold in `robustness.py` | n/a | Deterministic from group assignment |
