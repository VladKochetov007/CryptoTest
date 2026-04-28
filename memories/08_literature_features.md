---
name: literature_features
description: Feature ideas from DeFi rug-pull / pump-dump detection literature. Not yet implemented.
type: project
---

# Literature: DeFi Scam / Pump-Dump Detection Features

**Why:** Arxiv rate-limited (HTTP 429 on 2026-04-28). Summary from known literature.

## Key Papers & Feature Themes

### 1. Token-level static features at launch
Source: "SolRugDetector" (Solana-specific, 2024), Ethereum token scam literature.

- **Mint authority / freeze authority** at launch → honeypot indicator. Not in our dataset.
- **LP lock status** (liquidity locked vs not) → primary rug signal. Not in our dataset.
- **Initial liquidity amount in SOL** relative to market cap → skin-in-game proxy.
  We have `deployer_deposit_amount` which captures similar signal.
- **Top-10 holder concentration (HHI / Gini)** at slot 0. Not in dataset (only holders_count).
- **Dev wallet token allocation %** at launch. Not in dataset.

### 2. Network / graph features
Source: Ethereum wash-trade literature, "Token Spam Detection" papers.

- **Funding-source graph degree**: hub wallets that fund >50 deployers → rug factory.
  Approximated by `funder_prior_n` (existing) and `funder_prior_n_24h` (new).
- **Co-deployment clustering**: deployers funded by same parent wallet → linked.
  Not computable from current dataset without graph structure.
- **Cross-token buyer overlap**: wallets that appear in multiple tokens in same block.
  Not in dataset (no per-buyer wallet data).

### 3. Text / metadata features
Source: "Detecting fraudulent tokens using metadata" (Ethereum, 2023).

Key finding: simple regex over name/description outperforms ML on this task because
the signal is in the vocabulary, not the semantics. Key discriminators:
- All-caps tickers (DOGE-style) → higher hit rate for meme plays.
- Template descriptions ("deployed using X") → factory indicator.
- URLs in description → mixed signal (legitimate projects also link).
- Emoji in name → slightly positive for meme category but noisy.
- Social links count (twitter + telegram + website) → **negative** for hit_2x (our finding confirms).

### 4. Temporal / time-series features
Source: "Pump-and-dump in Ethereum token markets" (Xu & Livshits, 2019).

Key finding: regime features matter more than token-level fundamentals.
- **Hour-of-day** and **day-of-week** effects are strong.
- **Market-wide deployment rate** as a proxy for "hot market": more deploys per hour = lower
  quality per deploy (factories ramp up). We have `deploys_prev_15m` and `deploys_prev_60m`.
- **Relative token age at peak**: most P&D peaks within 3 minutes of launch.
  Matches our `deployer_first_sell_sec` p50=21s finding.

### 5. Buyer-side features (not computable from current dataset)
Source: "On the (in)security of DeFi protocols" (2022), Flashbots research.

- **Sniper bot count at slot 0**: JitoBundle detector. Requires per-buyer wallet data.
- **Top wallet PnL history**: requires cross-token wallet tracking.
- **MEV bot presence**: requires tx-level block data.
- `top_wallet_bought` (existing) is a coarse proxy for the above.

### 6. Cross-target label transfer
No direct literature found. User's prior experience: training on 5x improves 2x lift.
Theoretical basis: harder labels select only tokens with genuine momentum mechanics,
filtering out short-term bounces. Capturing "true rockets" vs "dead-cat bounces."

## Features Already Implemented That Match Literature
| Literature feature | Our implementation |
|---|---|
| Deployer launch frequency | `deployer_prior_n`, `deployer_prior_n_{1h,6h,24h,7d}` |
| Funding source hub | `funder_prior_n`, `funder_prior_n_24h` |
| Template metadata | `desc_has_deployed_template`, `desc_template_score` |
| Social link count | `has_twitter`, `has_website`, `has_telegram` (reversed sign) |
| Market deployment rate | `deploys_prev_15m`, `deploys_prev_60m` |
| Hour-of-day effects | `utc_hour`, `ny_hour`, `ldn_hour`, `tokyo_hour` |
| Macro crypto regime | `btc_close`, `sol_vol_1h`, `sol_ret_1h` |
| Image clone detection | `image_hash_seen_total` |
| Ticker collision | `same_ticker_today_prev` |

## Features Missing (require additional data)
| Feature | Data needed | Priority |
|---|---|---|
| Top-10 holder Gini/HHI | Helius `getTokenLargestAccounts` | High |
| LP lock status | Raydium/Orca LP events | High |
| Sniper count slot 0-3 | Per-buyer wallet trace | Medium |
| MEV bot presence | Block-level tx data | Medium |
| Funding graph degree | Source wallet graph | Medium |
| Cross-token buyer overlap | All-token wallet trace | Low |
