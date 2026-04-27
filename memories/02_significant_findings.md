# Significant Findings

## Features That Actually Matter (SHAP, LGBM instant model)

| Rank | Feature | Mean |SHAP| | Direction | Intuition |
|---|---|---|---|---|
| 1 | `deployer_prior_hit20k` | 0.558 | + | Deployer's past wins predict future wins. Track record compounds. |
| 2 | `deployer_seconds_since_last` | 0.435 | − | Very recent prior deploy → factory activity → bearish. Long gap → rarer, genuine. |
| 3 | `deployer_prior_n` | 0.375 | − (non-linear) | Few prior = genuine. Many = factory. But **not monotone**: optimal is ~1-5 prior tokens, not zero. |
| 4 | `deployer_wallet_balance_after_sol` | 0.182 | + | Higher balance = more committed wallet. |
| 5 | `image_hash_seen_total` | 0.167 | − | Clone image = copycat = lower quality. |
| 6 | `funder_prior_n` | 0.155 | − | Source wallet that funds many deployers = hub factory. |
| 7 | `mint_suffix_pump` | 0.143 | non-linear | Address ends in "pump" — correlated with certain factory styles. |
| 8 | `deployer_deposit_amount` | 0.143 | + | More SOL committed = more skin in game. |
| 9 | `name_upper_chars` | 0.128 | mixed | All-caps names (DOGE-style) have distinct hit-rate profile. |
| 10 | `btc_close` | 0.108 | + | BTC level = macro regime. Bull regime → more buyers of anything. |

Features outside top 10 with solid contribution: `utc_hour`, `ny_hour`, `sol_vol_1h`,
`deployer_prior_grad`, `same_ticker_today_prev`, `has_twitter` (reversed), `desc_len`.

## Confirmed Insignificant Findings

| Feature / Heuristic | IV / AUC | Notes |
|---|---|---|
| `is_cex` (boolean) | IV = 0.0001, AUC ≈ 0.500 | **Completely useless**. Task's "+20 if CEX" is fiction. |
| CEX-funded vs non-CEX | hit_20k 2.30% vs 2.21% | Difference is not statistically significant at this scale. |
| `ticker_len` | Spearman ρ = −0.003 | Noise. |
| `has_telegram` | IV = 0.0003 | Negligible. |

## Reversed-Sign Features (expected +, actual −)

| Feature | Expected | Actual |
|---|---|---|
| `has_twitter` | + (shows legitimacy) | **−**: no-twitter rate 18.4% vs with-twitter 14.8%. Bots add socials, real projects don't need them. |
| `has_desc` | + | **−**: same pattern. Description padding is a factory spam signal. |
| `deployer_prior_n = 0` | + (fresh wallet = not a rug factory) | **~neutral**: SHAP shows non-monotone. First-timers are neither better nor worse; experienced deployers with WINS are best. |
| `same_ticker_today_prev` | − expected (clone = bad) | Forward but weak, non-monotone. Some ticker collisions are just popular themes, not scams. |

## Time Stability — Regime Risk

Hit_2x base rate across 10 chronological chunks: min 12.4%, max 19.0%, std 2.2%.
Relative range = **+53%**. A static threshold calibrated on early data would be
systematically mis-calibrated on later data.

**Action required**: production must recalibrate isotonic regression and decision
thresholds on a rolling 7-day window, not once at train time.

## CEX Identity vs CEX Boolean

The CEX *name* (categorical WoE) carries IV = 0.078 vs boolean IV = 0.0001.
Per-exchange hit_2x rates:
- Gate.io: 22.5% (lift 1.41)
- KuCoin: 18.2% (lift 1.14)
- Coinbase: 17.2% (lift 1.08)
- Binance: 14.6% (lift 0.91)
- MEXC: 8.0% (lift 0.50) — strong **negative** signal

## Macro Context (SOL/BTC via ccxt)

BTC price level (`btc_close`) ranks #10 in SHAP. SOL volatility `sol_vol_1h` also
contributes. In bull regimes more buyers pile into everything; in bear regimes even
good tokens die. These 8 market-context features cost ~15 seconds of API latency on
first boot (not hot-path — pre-fetched and joined at deploy time).

## Deployer-Grouped CV — Alpha Source Confirmed

GroupKFold(n_splits=5) on `deployer_address` — no deployer in train+test:
- Mean AUC = **0.778**, better than time-fold 0.765
- Confirms: model learns *feature patterns*, not deployer identities
- The top deployer (8,465 tokens) inflates the time-fold test set's regime composition
  but does not inflate AUC through memorisation

## Two-Stage Architecture — Lift vs Single-Stage

| Stage | AUC | Use |
|---|---|---|
| Deploy-time instant | 0.765 | Gate: buy probe or skip entirely |
| Post-60-s with60s | 0.945 | Add: scale up or abort probe |
| Combined policy | — | Product: 0.1 SOL probe → 1.0 SOL or exit at 60 s |

The 0.945 with60s AUC justifies the probe cost: expected value of aborting bad probes
at 60 s (16.5% of buys) is approximately `0.165 × 0.5 SOL × abort_lift` SOL saved
per trade cycle.
