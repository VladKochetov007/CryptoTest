"""Pump.fun 500k EDA — pre-buy scoring + exit strategy framing.

Run:  uv run --with marimo marimo edit eda/notebook.py
Or:   .venv/bin/marimo edit eda/notebook.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    DATA = Path("/home/vlad/development/PumpTest")
    TOKENS = DATA / "tokens.parquet"
    SLOTS = DATA / "slot_features_60m.parquet"
    ACTS = DATA / "deployer_actions_60m.parquet"
    PLOTS = DATA / "eda" / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW tokens AS SELECT * FROM read_parquet('{TOKENS}')")
    con.execute(f"CREATE VIEW slots  AS SELECT * FROM read_parquet('{SLOTS}')")
    con.execute(f"CREATE VIEW acts   AS SELECT * FROM read_parquet('{ACTS}')")
    return DATA, PLOTS, TOKENS, SLOTS, ACTS, con, duckdb, pl, np, plt, Path


@app.cell
def _(con):
    counts = con.sql("""
        SELECT 'tokens' t, COUNT(*) n FROM tokens
        UNION ALL SELECT 'slots',   COUNT(*) FROM slots
        UNION ALL SELECT 'actions', COUNT(*) FROM acts
    """).pl()
    counts
    return counts,


@app.cell
def _(con):
    # null rates on key tokens columns
    nulls = con.sql("""
        SELECT
          AVG(CASE WHEN image_hash_sha256 IS NULL THEN 1 ELSE 0 END) img_null,
          AVG(CASE WHEN description IS NULL THEN 1 ELSE 0 END) desc_null,
          AVG(CASE WHEN website IS NULL THEN 1 ELSE 0 END) web_null,
          AVG(CASE WHEN twitter_handle IS NULL THEN 1 ELSE 0 END) tw_null,
          AVG(CASE WHEN telegram_link IS NULL THEN 1 ELSE 0 END) tg_null,
          AVG(CASE WHEN deployer_deposit_amount IS NULL THEN 1 ELSE 0 END) dep_null,
          AVG(CASE WHEN deployer_wallet_source IS NULL THEN 1 ELSE 0 END) src_null,
          AVG(CASE WHEN deployer_wallet_source_is_cex THEN 1 ELSE 0 END) cex_share
        FROM tokens
    """).pl()
    nulls
    return nulls,


@app.cell
def _(con, plt, PLOTS):
    # ATH mcap heavy-tail
    ath = con.sql("SELECT ath_market_cap_usd FROM tokens WHERE ath_market_cap_usd > 0").pl().to_numpy().ravel()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ath, bins=200)
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_title("ATH market cap (USD) — log-log")
    ax[0].set_xlabel("USD"); ax[0].set_ylabel("# tokens")
    # CDF
    a = ath[ath > 0]
    a = sorted(a)
    import numpy as np
    cdf = np.arange(1, len(a) + 1) / len(a)
    ax[1].plot(a, 1 - cdf)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_title("Survival: P(ATH >= x)")
    ax[1].set_xlabel("USD"); ax[1].set_ylabel("P")
    fig.tight_layout()
    fig.savefig(PLOTS / "ath_mcap.png", dpi=110)
    fig
    return ath, fig


@app.cell
def _(con):
    # graduation thresholds — pump.fun graduates at ~$69k mcap
    grads = con.sql("""
        SELECT
          AVG(CASE WHEN ath_market_cap_usd>=20000  THEN 1 ELSE 0 END) p20k,
          AVG(CASE WHEN ath_market_cap_usd>=69000  THEN 1 ELSE 0 END) p69k_grad,
          AVG(CASE WHEN ath_market_cap_usd>=100000 THEN 1 ELSE 0 END) p100k,
          AVG(CASE WHEN ath_market_cap_usd>=500000 THEN 1 ELSE 0 END) p500k,
          AVG(CASE WHEN ath_market_cap_usd>=1e6    THEN 1 ELSE 0 END) p1m
        FROM tokens
    """).pl()
    grads
    return grads,


@app.cell
def _(con, plt, PLOTS):
    # deployer concentration — power law
    dep = con.sql("""
        SELECT n_tokens, COUNT(*) AS deployers FROM (
            SELECT deployer_address, COUNT(*) AS n_tokens FROM tokens GROUP BY 1
        ) GROUP BY 1 ORDER BY n_tokens
    """).pl()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(dep["n_tokens"].to_numpy(), dep["deployers"].to_numpy(), s=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tokens per deployer"); ax.set_ylabel("# deployers")
    ax.set_title("Deployer concentration (Pareto check)")
    fig.tight_layout()
    fig.savefig(PLOTS / "deployer_concentration.png", dpi=110)
    fig
    return dep, fig


@app.cell
def _(con, plt, PLOTS):
    # deposit_amount log dist
    deposits = con.sql("""
        SELECT deployer_deposit_amount FROM tokens
        WHERE deployer_deposit_amount > 0
    """).pl().to_numpy().ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deposits, bins=200)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Deployer deposit (SOL)")
    ax.set_xlabel("SOL"); ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(PLOTS / "deployer_deposit.png", dpi=110)
    fig
    return deposits, fig


@app.cell
def _(con):
    # CEX heuristic check vs hit_20k — CRITICAL
    cex_check = con.sql("""
        WITH a AS (
            SELECT
              deployer_wallet_source_is_cex AS cex,
              CASE WHEN deployer_wallet_source IS NULL THEN 'unknown'
                   WHEN deployer_wallet_source_is_cex THEN deployer_wallet_source_cex_name
                   ELSE 'other_wallet' END AS bucket,
              CASE WHEN ath_market_cap_usd>=20000 THEN 1 ELSE 0 END AS hit20k,
              CASE WHEN ath_market_cap_usd>=69000 THEN 1 ELSE 0 END AS hit_grad
            FROM tokens
        )
        SELECT bucket, COUNT(*) n, AVG(hit20k) hit_20k, AVG(hit_grad) hit_grad
        FROM a GROUP BY 1 ORDER BY n DESC
    """).pl()
    cex_check
    return cex_check,


@app.cell
def _(con):
    # Image hash duplication — copycat detection
    img_dup = con.sql("""
        WITH a AS (SELECT image_hash_sha256, COUNT(*) AS n
                   FROM tokens WHERE image_hash_sha256 IS NOT NULL GROUP BY 1)
        SELECT
          AVG(CASE WHEN n=1 THEN 1 ELSE 0 END) frac_unique,
          AVG(CASE WHEN n>1 THEN n ELSE 0 END) avg_dup_size,
          MAX(n) most_reused,
          COUNT(*) FILTER(WHERE n>=10) hashes_used_10plus,
          (SELECT COUNT(*) FROM tokens WHERE image_hash_sha256 IN
              (SELECT image_hash_sha256 FROM a WHERE n>=10)) tokens_in_big_clones
        FROM a
    """).pl()
    img_dup
    return img_dup,


@app.cell
def _(con):
    # name/ticker collision per day
    name_clash = con.sql("""
        WITH a AS (
            SELECT date_trunc('day', deploy_time_utc) AS d, ticker, COUNT(*) AS n
            FROM tokens GROUP BY 1, 2
        )
        SELECT
          AVG(CASE WHEN n>1 THEN 1 ELSE 0 END) frac_collision,
          MAX(n) max_same_ticker_day
        FROM a
    """).pl()
    name_clash
    return name_clash,


@app.cell
def _(con, plt, PLOTS):
    # Survival curve: % tokens still trading at second t
    surv = con.sql("""
        WITH bucket AS (
            SELECT token_id, MAX(seconds_since_deploy) AS last_sec
            FROM slots GROUP BY 1
        )
        SELECT s, AVG(CASE WHEN last_sec>=s THEN 1 ELSE 0 END) frac_alive
        FROM bucket, generate_series(0, 3600, 60) g(s) GROUP BY s ORDER BY s
    """).pl()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(surv["s"].to_numpy() / 60, surv["frac_alive"].to_numpy())
    ax.set_xlabel("minutes since deploy"); ax.set_ylabel("frac with trades >= t")
    ax.set_title("Token activity survival in first 60min")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "survival.png", dpi=110)
    fig
    return surv, fig


@app.cell
def _(con):
    # 30min ROI label distribution — for Part1 target
    roi30 = con.sql("""
        WITH first AS (
            SELECT token_id, MIN(seconds_since_deploy) AS s0
            FROM slots WHERE price_sol_per_token>0 GROUP BY 1
        ),
        p0 AS (
            SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0
            FROM slots s JOIN first f ON s.token_id=f.token_id AND s.seconds_since_deploy=f.s0
            WHERE s.price_sol_per_token>0 GROUP BY 1
        ),
        peak AS (
            SELECT token_id, MAX(price_sol_per_token) AS pmax
            FROM slots WHERE seconds_since_deploy<=1800 AND price_sol_per_token>0 GROUP BY 1
        )
        SELECT
          COUNT(*) n,
          AVG(CASE WHEN pmax/p0>=1.5 THEN 1 ELSE 0 END) base_1_5x,
          AVG(CASE WHEN pmax/p0>=2   THEN 1 ELSE 0 END) base_2x,
          AVG(CASE WHEN pmax/p0>=3   THEN 1 ELSE 0 END) base_3x,
          AVG(CASE WHEN pmax/p0>=5   THEN 1 ELSE 0 END) base_5x,
          AVG(CASE WHEN pmax/p0>=10  THEN 1 ELSE 0 END) base_10x,
          approx_quantile(pmax/p0, 0.5) p50_roi,
          approx_quantile(pmax/p0, 0.9) p90_roi,
          approx_quantile(pmax/p0, 0.99) p99_roi
        FROM p0 JOIN peak USING (token_id)
    """).pl()
    roi30
    return roi30,


@app.cell
def _(con):
    # deposit_amount bin → hit_2x label (use first-slot anchor)
    bin_check = con.sql("""
        WITH first AS (
            SELECT token_id, MIN(seconds_since_deploy) AS s0
            FROM slots WHERE price_sol_per_token>0 GROUP BY 1
        ),
        p0 AS (
            SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0
            FROM slots s JOIN first f ON s.token_id=f.token_id AND s.seconds_since_deploy=f.s0
            WHERE s.price_sol_per_token>0 GROUP BY 1
        ),
        peak AS (
            SELECT token_id, MAX(price_sol_per_token) AS pmax
            FROM slots WHERE seconds_since_deploy<=1800 AND price_sol_per_token>0 GROUP BY 1
        ),
        lab AS (
            SELECT token_id, CASE WHEN pmax/p0>=2 THEN 1 ELSE 0 END AS hit_2x
            FROM p0 JOIN peak USING (token_id)
        )
        SELECT
          CASE
            WHEN t.deployer_deposit_amount<=0.2 THEN '0_0.2'
            WHEN t.deployer_deposit_amount<=1   THEN '0.2_1'
            WHEN t.deployer_deposit_amount<=5   THEN '1_5'
            WHEN t.deployer_deposit_amount<=20  THEN '5_20'
            ELSE '20+' END AS bin,
          COUNT(*) n, AVG(lab.hit_2x) base_rate
        FROM tokens t JOIN lab USING (token_id)
        WHERE t.deployer_deposit_amount IS NOT NULL
        GROUP BY 1 ORDER BY base_rate DESC
    """).pl()
    bin_check
    return bin_check,


@app.cell
def _(con):
    # deployer prior-token count vs hit_2x — heuristic test (factory wallets)
    prior = con.sql("""
        WITH ranked AS (
            SELECT token_id, deployer_address,
                   ROW_NUMBER() OVER (PARTITION BY deployer_address ORDER BY deploy_time_unix) - 1 AS prior_n
            FROM tokens
        ),
        first AS (SELECT token_id, MIN(seconds_since_deploy) AS s0 FROM slots WHERE price_sol_per_token>0 GROUP BY 1),
        p0 AS (SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0
               FROM slots s JOIN first f ON s.token_id=f.token_id AND s.seconds_since_deploy=f.s0
               WHERE s.price_sol_per_token>0 GROUP BY 1),
        peak AS (SELECT token_id, MAX(price_sol_per_token) AS pmax
                 FROM slots WHERE seconds_since_deploy<=1800 AND price_sol_per_token>0 GROUP BY 1),
        lab AS (SELECT token_id, CASE WHEN pmax/p0>=2 THEN 1 ELSE 0 END AS hit_2x FROM p0 JOIN peak USING (token_id))
        SELECT
          CASE
            WHEN prior_n=0 THEN 'first'
            WHEN prior_n<=2 THEN '1-2'
            WHEN prior_n<=10 THEN '3-10'
            WHEN prior_n<=50 THEN '11-50'
            ELSE '50+' END AS prior_bucket,
          COUNT(*) n, AVG(lab.hit_2x) hit_2x
        FROM ranked r JOIN lab USING (token_id) GROUP BY 1
    """).pl()
    prior
    return prior,


@app.cell
def _(con):
    # deployer first-sell timing — exit signal source (rug detection)
    first_sell = con.sql("""
        WITH d AS (
            SELECT token_id, MIN(seconds_since_deploy) AS first_sell_sec
            FROM acts WHERE deployer_action ILIKE '%sell%' GROUP BY 1
        )
        SELECT
          COUNT(*) n_with_dep_sell,
          approx_quantile(first_sell_sec, 0.25) p25,
          approx_quantile(first_sell_sec, 0.5)  p50,
          approx_quantile(first_sell_sec, 0.9)  p90,
          AVG(CASE WHEN first_sell_sec<=60 THEN 1 ELSE 0 END) sell_le60s
        FROM d
    """).pl()
    first_sell
    return first_sell,


@app.cell
def _(con, plt, PLOTS):
    # Median early volume curve, winners (hit_2x) vs losers
    curves = con.sql("""
        WITH first AS (SELECT token_id, MIN(seconds_since_deploy) AS s0 FROM slots WHERE price_sol_per_token>0 GROUP BY 1),
        p0 AS (SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0 FROM slots s JOIN first f ON s.token_id=f.token_id AND s.seconds_since_deploy=f.s0 WHERE s.price_sol_per_token>0 GROUP BY 1),
        peak AS (SELECT token_id, MAX(price_sol_per_token) AS pmax FROM slots WHERE seconds_since_deploy<=1800 AND price_sol_per_token>0 GROUP BY 1),
        lab AS (SELECT token_id, CASE WHEN pmax/p0>=2 THEN 1 ELSE 0 END AS w FROM p0 JOIN peak USING (token_id)),
        bucket AS (
            SELECT FLOOR(seconds_since_deploy/15)*15 AS sec_bin, lab.w,
                   approx_quantile(volume_sol, 0.5) AS vol_med,
                   approx_quantile(holders_count, 0.5) AS h_med,
                   COUNT(*) n
            FROM slots JOIN lab USING (token_id)
            WHERE seconds_since_deploy<=900
            GROUP BY 1, 2
        )
        SELECT * FROM bucket ORDER BY sec_bin, w
    """).pl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for w, label, c in [(0, "loser", "C0"), (1, "winner_2x", "C1")]:
        sub = curves.filter(curves["w"] == w)
        ax[0].plot(sub["sec_bin"].to_numpy(), sub["vol_med"].to_numpy(), label=label, color=c)
        ax[1].plot(sub["sec_bin"].to_numpy(), sub["h_med"].to_numpy(), label=label, color=c)
    ax[0].set_title("Median per-slot volume (SOL)"); ax[0].legend(); ax[0].set_xlabel("sec since deploy")
    ax[1].set_title("Median holders count"); ax[1].legend(); ax[1].set_xlabel("sec since deploy")
    fig.tight_layout()
    fig.savefig(PLOTS / "winner_vs_loser_curves.png", dpi=110)
    fig
    return curves, fig


@app.cell
def _(con):
    # top wallet activity early vs ROI (predictive signal probe)
    topw = con.sql("""
        WITH first60 AS (
            SELECT token_id, MAX(CASE WHEN top_wallet_bought THEN 1 ELSE 0 END) AS smart_in_60
            FROM slots WHERE seconds_since_deploy<=60 GROUP BY 1
        ),
        first AS (SELECT token_id, MIN(seconds_since_deploy) AS s0 FROM slots WHERE price_sol_per_token>0 GROUP BY 1),
        p0 AS (SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0 FROM slots s JOIN first f ON s.token_id=f.token_id AND s.seconds_since_deploy=f.s0 WHERE s.price_sol_per_token>0 GROUP BY 1),
        peak AS (SELECT token_id, MAX(price_sol_per_token) AS pmax FROM slots WHERE seconds_since_deploy<=1800 AND price_sol_per_token>0 GROUP BY 1),
        lab AS (SELECT token_id, CASE WHEN pmax/p0>=2 THEN 1 ELSE 0 END AS hit_2x FROM p0 JOIN peak USING (token_id))
        SELECT smart_in_60, COUNT(*) n, AVG(hit_2x) hit_2x
        FROM first60 JOIN lab USING (token_id) GROUP BY 1
    """).pl()
    topw
    return topw,


@app.cell
def _(con):
    # graduation rate (using ATH ~ $69k threshold)
    grad_rate = con.sql("""
        WITH d AS (
            SELECT date_trunc('day', deploy_time_utc) AS d,
                   AVG(CASE WHEN ath_market_cap_usd>=69000 THEN 1 ELSE 0 END) grad_rate,
                   COUNT(*) n
            FROM tokens GROUP BY 1
        ) SELECT * FROM d ORDER BY d
    """).pl()
    grad_rate
    return grad_rate,


@app.cell
def _():
    import marimo as mo
    mo.md("""
# Findings (caveman)

* **500k tokens, 91k deployers**. Top single deployer made 8465 tokens (factory).
* **CEX-funded share = 3.5%** (not 50% like task hint suggests). And CEX vs non-CEX hit_20k base rate **identical at ~2.3%** — the canonical heuristic does NOT discriminate in this dataset. Save +20 points for something real.
* **Deposit amount IS predictive**: tiny (≤0.2 SOL) → 1.5% hit_20k. ≥20 SOL → 4.2% hit_20k. Monotone bin trend. Use as continuous feature, not boolean.
* **ATH heavy tail**: p50 = $4.3k, p99 = $37k. Graduation (~$69k) happens to ~0.5% of tokens.
* **Survival**: 57% trade past 60s, 14% past 20min, 4% past 50min. Most tokens DEAD within minutes.
* **Hit_2x in 30min from first-slot price = 15.2%** — clean, well-balanced label for ML.
* **Image hash dups exist** (need feature: is_clone).
* **Ticker collisions per day common** (top duplicates many).
* **First-sell deployer action timing** is core exit signal data.

## Modeling decisions

**Part 1 (pre-buy ≤200ms)**: only deploy-time features allowed (tokens table only). Time-based split (train early, test late). Label = hit_2x_30min anchored at first-slot price. Class balance ~15%, fine for XGBoost. Deploy-time features:
- log(deposit_amount), balance_before, source_amount, is_CEX (low-info), CEX one-hot
- deployer_prior_token_count (count up to deploy_time)
- deployer_prior_grad_count (factory quality)
- name/ticker length, has_website, has_twitter, has_telegram, has_description
- ticker_collision_today, image_hash_seen_count (clone score)
- minute_of_day, day_of_week (regime)

**Part 2 (exit)**: 100+ buys → simulate via slot_features. Strategies:
1. fixed 2x take-profit + 50% stop
2. trailing stop on max drawdown
3. exit on top_wallet_bought=False & volume drops 80%
4. exit on first deployer sell (`acts` table)
5. exit on volume stagnation N consecutive slots
Backtest with vectorbt or pure polars. Report median ROI, max drawdown, hold time, hit ratio.

## Leakage red flags
- ATH and latest_market_cap_usd are FUTURE — drop from features, use only as label source.
- deployer_prior_* features must use `deploy_time_unix` cutoff strictly (no leak).
- Slot-feature engineering for Part 2 must respect cumulative time only.
""")
    return mo,


if __name__ == "__main__":
    app.run()
