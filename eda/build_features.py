"""Feature engineering for pre-buy scoring (Part 1) and exit backtesting (Part 2).

Features split into:
- DEPLOY-TIME (Part 1, available before slot 0): deployer history, funding, metadata,
  cross-sectional regime, time-of-day, market context (SOL OHLCV via ccxt).
- POST-DEPLOY (Part 2 / 2-stage): early slot aggregates, deployer first-action timing,
  trade-side dynamics, holders growth, sniper concentration proxy.

All deployer-history features use STRICT <= time-of-deploy ordering (no leakage).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import duckdb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "tokens.parquet"
SLOTS = ROOT / "slot_features_60m.parquet"
ACTS = ROOT / "deployer_actions_60m.parquet"
OUT = ROOT / "eda"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# market context (SOL/BTC) via ccxt — append per-token rolling vol features
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, since_ms: int, until_ms: int, timeframe: str = "5m") -> pl.DataFrame:
    """Fetch ccxt OHLCV in a date range. Paginates through Binance limits."""
    ex = ccxt.binance()
    rows = []
    cursor = since_ms
    while cursor < until_ms:
        chunk = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        last = chunk[-1][0]
        if last <= cursor:
            break
        cursor = last + 1
    return pl.DataFrame(rows, schema=["ts_ms", "open", "high", "low", "close", "volume"], orient="row")


def market_context(deploy_min_unix: int, deploy_max_unix: int) -> pl.DataFrame:
    """Build one row per 5-minute slot covering the dataset, with SOL/BTC features."""
    pad_before = 24 * 3600
    since = (deploy_min_unix - pad_before) * 1000
    until = (deploy_max_unix + 3600) * 1000
    sol = fetch_ohlcv("SOL/USDT", since, until, "5m").rename({c: f"sol_{c}" for c in ["open", "high", "low", "close", "volume"]})
    btc = fetch_ohlcv("BTC/USDT", since, until, "5m").rename({c: f"btc_{c}" for c in ["open", "high", "low", "close", "volume"]})
    df = sol.join(btc, on="ts_ms", how="inner")

    df = df.sort("ts_ms").with_columns(
        (pl.col("sol_close").pct_change()).alias("sol_ret_5m"),
        (pl.col("btc_close").pct_change()).alias("btc_ret_5m"),
    ).with_columns(
        pl.col("sol_ret_5m").rolling_std(window_size=12, min_periods=6).alias("sol_vol_1h"),
        pl.col("sol_ret_5m").rolling_std(window_size=288, min_periods=48).alias("sol_vol_24h"),
        pl.col("btc_ret_5m").rolling_std(window_size=12, min_periods=6).alias("btc_vol_1h"),
        (pl.col("sol_close").pct_change(12)).alias("sol_ret_1h"),
        (pl.col("sol_close").pct_change(288)).alias("sol_ret_24h"),
        (pl.col("btc_close").pct_change(12)).alias("btc_ret_1h"),
    )
    return df.select(
        (pl.col("ts_ms") // 1000).alias("ts_unix"),
        "sol_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "sol_ret_24h",
        "btc_close", "btc_vol_1h", "btc_ret_1h",
    )


# ---------------------------------------------------------------------------
# deployer history (rolling, no-leak)
# ---------------------------------------------------------------------------

def deployer_history_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """For each (deployer, token), prior_n, prior_grad_count, time_since_last."""
    return con.sql(
        """
        WITH t AS (
            SELECT token_id, deployer_address, deploy_time_unix,
                   ath_market_cap_usd >= 69000 AS hit_grad,
                   ath_market_cap_usd >= 20000 AS hit_20k
            FROM tokens
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY deployer_address ORDER BY deploy_time_unix) - 1
                    AS deployer_prior_n,
                LAG(deploy_time_unix) OVER (PARTITION BY deployer_address ORDER BY deploy_time_unix)
                    AS prev_deploy_time
            FROM t
        ),
        cum AS (
            SELECT *,
                COALESCE(SUM(CASE WHEN hit_grad THEN 1 ELSE 0 END) OVER
                    (PARTITION BY deployer_address ORDER BY deploy_time_unix
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS deployer_prior_grad,
                COALESCE(SUM(CASE WHEN hit_20k THEN 1 ELSE 0 END) OVER
                    (PARTITION BY deployer_address ORDER BY deploy_time_unix
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS deployer_prior_hit20k
            FROM ranked
        )
        SELECT token_id, deployer_prior_n, deployer_prior_grad,
               deployer_prior_hit20k,
               COALESCE(deploy_time_unix - prev_deploy_time, 86400) AS deployer_seconds_since_last
        FROM cum
        """
    ).pl()


def funder_history_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """How often does the source wallet fund deployers, and how many of those produce hits."""
    return con.sql(
        """
        WITH t AS (
            SELECT token_id, deployer_wallet_source AS src, deploy_time_unix,
                   ath_market_cap_usd >= 20000 AS hit_20k,
                   ath_market_cap_usd >= 69000 AS hit_grad
            FROM tokens
            WHERE deployer_wallet_source IS NOT NULL
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY src ORDER BY deploy_time_unix) - 1 AS funder_prior_n,
                COALESCE(SUM(CASE WHEN hit_20k THEN 1 ELSE 0 END) OVER
                    (PARTITION BY src ORDER BY deploy_time_unix
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS funder_prior_hit20k,
                COALESCE(SUM(CASE WHEN hit_grad THEN 1 ELSE 0 END) OVER
                    (PARTITION BY src ORDER BY deploy_time_unix
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS funder_prior_grad
            FROM t
        )
        SELECT token_id, funder_prior_n, funder_prior_hit20k, funder_prior_grad
        FROM ranked
        """
    ).pl()


# ---------------------------------------------------------------------------
# cross-sectional regime — strictly past lookback
# ---------------------------------------------------------------------------

def regime_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Rolling counts of recent deploys + recent hit rate using time-bucket windows.

    Strategy: bucket deploy_time_unix to 1-minute, count deploys per minute, then
    use rolling sums over the past N minutes — each token reads only past buckets.
    """
    df = con.sql(
        """
        SELECT token_id, deploy_time_unix,
               (deploy_time_unix / 60)::BIGINT AS min_bucket,
               CASE WHEN ath_market_cap_usd >= 20000 THEN 1 ELSE 0 END AS hit_20k
        FROM tokens
        """
    ).pl().sort("deploy_time_unix")

    per_min = (
        df.group_by("min_bucket")
        .agg(pl.len().alias("deploys_per_min"), pl.col("hit_20k").sum().alias("hits_per_min"))
        .sort("min_bucket")
    )
    per_min = per_min.with_columns(
        per_min["deploys_per_min"].rolling_sum(window_size=15, min_periods=1).alias("deploys_15m"),
        per_min["deploys_per_min"].rolling_sum(window_size=60, min_periods=1).alias("deploys_60m"),
        per_min["hits_per_min"].rolling_sum(window_size=60, min_periods=1).alias("hits_60m"),
    ).with_columns(
        (pl.col("hits_60m") / pl.col("deploys_60m").clip(lower_bound=1)).alias("hit20k_rate_60m"),
    )
    # shift by one row so token at minute m sees windows ending at m-1
    per_min = per_min.with_columns(
        pl.col("deploys_15m").shift(1).alias("deploys_prev_15m"),
        pl.col("deploys_60m").shift(1).alias("deploys_prev_60m"),
        pl.col("hit20k_rate_60m").shift(1).alias("hit20k_rate_prev_60m"),
    )
    return df.join(per_min.select("min_bucket", "deploys_prev_15m", "deploys_prev_60m", "hit20k_rate_prev_60m"),
                   on="min_bucket", how="left").select(
        "token_id", "deploys_prev_15m", "deploys_prev_60m", "hit20k_rate_prev_60m"
    )


# ---------------------------------------------------------------------------
# clones / sybil-style features
# ---------------------------------------------------------------------------

def clone_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.sql(
        """
        WITH img AS (
            SELECT image_hash_sha256, COUNT(*) AS hash_seen FROM tokens
            WHERE image_hash_sha256 IS NOT NULL GROUP BY 1
        ),
        ticker_day AS (
            SELECT t.token_id, COUNT(*) - 1 AS same_ticker_today
            FROM tokens t JOIN tokens t2
              ON t.ticker = t2.ticker
             AND date_trunc('day', t.deploy_time_utc) = date_trunc('day', t2.deploy_time_utc)
             AND t.token_id <> t2.token_id
             AND t2.deploy_time_unix < t.deploy_time_unix
            GROUP BY t.token_id
        ),
        name_hour AS (
            SELECT t.token_id, COUNT(*) AS same_name_hour
            FROM tokens t JOIN tokens t2
              ON LOWER(t.name) = LOWER(t2.name)
             AND t.token_id <> t2.token_id
             AND t2.deploy_time_unix BETWEEN t.deploy_time_unix - 3600 AND t.deploy_time_unix - 1
            GROUP BY t.token_id
        )
        SELECT t.token_id,
               COALESCE(img.hash_seen, 1) AS image_hash_seen_total,
               COALESCE(tk.same_ticker_today, 0) AS same_ticker_today_prev,
               COALESCE(nm.same_name_hour, 0) AS same_name_prev_hour
        FROM tokens t
        LEFT JOIN img ON img.image_hash_sha256 = t.image_hash_sha256
        LEFT JOIN ticker_day tk USING (token_id)
        LEFT JOIN name_hour nm USING (token_id)
        """
    ).pl()


# ---------------------------------------------------------------------------
# address features (vanity, suffix == 'pump', etc.)
# ---------------------------------------------------------------------------

def address_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.sql(
        """
        SELECT token_id,
               CASE WHEN token_mint_address LIKE '%pump' THEN 1 ELSE 0 END AS mint_suffix_pump,
               CASE WHEN token_mint_address LIKE '%PUMP' THEN 1 ELSE 0 END AS mint_suffix_PUMP,
               CASE WHEN deployer_address LIKE '%pump' THEN 1 ELSE 0 END AS deployer_suffix_pump,
               LENGTH(REGEXP_REPLACE(LOWER(name), '[^a-z]', '', 'g')) AS name_alpha_chars,
               LENGTH(REGEXP_REPLACE(name, '[^A-Z]', '', 'g')) AS name_upper_chars
        FROM tokens
        """
    ).pl()


# ---------------------------------------------------------------------------
# post-deploy aggregates (for 2-stage / Part 2)
# ---------------------------------------------------------------------------

def post_deploy_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.sql(
        """
        WITH s30 AS (
            SELECT token_id,
                   SUM(buy_volume_sol)  AS buy_vol_30s,
                   SUM(sell_volume_sol) AS sell_vol_30s,
                   COUNT(*)             AS n_slots_30s,
                   MAX(holders_count)   AS holders_30s,
                   MAX(CASE WHEN top_wallet_bought THEN 1 ELSE 0 END) AS top_wallet_30s,
                   SUM(trade_count)     AS trades_30s
            FROM slots WHERE seconds_since_deploy <= 30 GROUP BY 1
        ),
        s60 AS (
            SELECT token_id,
                   SUM(buy_volume_sol) AS buy_vol_60s, SUM(sell_volume_sol) AS sell_vol_60s,
                   MAX(holders_count) AS holders_60s,
                   MAX(CASE WHEN top_wallet_bought THEN 1 ELSE 0 END) AS top_wallet_60s,
                   SUM(trade_count) AS trades_60s,
                   approx_quantile(price_sol_per_token, 0.99) AS price_max_60s,
                   approx_quantile(price_sol_per_token, 0.01) AS price_min_60s
            FROM slots WHERE seconds_since_deploy <= 60 AND price_sol_per_token > 0 GROUP BY 1
        ),
        s120 AS (
            SELECT token_id,
                   SUM(buy_volume_sol) AS buy_vol_120s, SUM(sell_volume_sol) AS sell_vol_120s,
                   MAX(holders_count) AS holders_120s,
                   COUNT(DISTINCT block_slot) AS uniq_slots_120s
            FROM slots WHERE seconds_since_deploy <= 120 GROUP BY 1
        ),
        first_sell AS (
            SELECT token_id, MIN(seconds_since_deploy) AS deployer_first_sell_sec
            FROM acts WHERE deployer_action ILIKE '%sell%' GROUP BY 1
        )
        SELECT
            t.token_id,
            COALESCE(s30.buy_vol_30s, 0) AS buy_vol_30s,
            COALESCE(s30.sell_vol_30s, 0) AS sell_vol_30s,
            COALESCE(s30.n_slots_30s, 0) AS active_slots_30s,
            COALESCE(s30.holders_30s, 0) AS holders_30s,
            COALESCE(s30.top_wallet_30s, 0) AS top_wallet_30s,
            COALESCE(s30.trades_30s, 0) AS trades_30s,
            COALESCE(s60.buy_vol_60s, 0) AS buy_vol_60s,
            COALESCE(s60.sell_vol_60s, 0) AS sell_vol_60s,
            COALESCE(s60.holders_60s, 0) AS holders_60s,
            COALESCE(s60.top_wallet_60s, 0) AS top_wallet_60s,
            COALESCE(s60.trades_60s, 0) AS trades_60s,
            s60.price_max_60s, s60.price_min_60s,
            COALESCE(s120.buy_vol_120s, 0) AS buy_vol_120s,
            COALESCE(s120.sell_vol_120s, 0) AS sell_vol_120s,
            COALESCE(s120.holders_120s, 0) AS holders_120s,
            COALESCE(s120.uniq_slots_120s, 0) AS uniq_slots_120s,
            first_sell.deployer_first_sell_sec
        FROM tokens t
        LEFT JOIN s30 USING (token_id)
        LEFT JOIN s60 USING (token_id)
        LEFT JOIN s120 USING (token_id)
        LEFT JOIN first_sell USING (token_id)
        """
    ).pl()


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

def label_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.sql(
        """
        WITH first AS (
            SELECT token_id, MIN(seconds_since_deploy) AS s0
            FROM slots WHERE price_sol_per_token > 0 GROUP BY 1
        ),
        p0 AS (
            SELECT s.token_id, ANY_VALUE(s.price_sol_per_token) AS p0,
                   ANY_VALUE(s.block_slot) AS first_block_slot
            FROM slots s JOIN first f
              ON s.token_id = f.token_id AND s.seconds_since_deploy = f.s0
            WHERE s.price_sol_per_token > 0 GROUP BY 1
        ),
        peak AS (
            SELECT token_id, MAX(price_sol_per_token) AS pmax_30m,
                   ARG_MAX(seconds_since_deploy, price_sol_per_token) AS peak_sec_30m
            FROM slots WHERE seconds_since_deploy <= 1800 AND price_sol_per_token > 0 GROUP BY 1
        )
        SELECT token_id, p0, pmax_30m, peak_sec_30m, first_block_slot
        FROM p0 JOIN peak USING (token_id)
        """
    ).pl()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tokens AS SELECT * FROM read_parquet('{TOKENS}')")
    con.execute(f"CREATE VIEW slots  AS SELECT * FROM read_parquet('{SLOTS}')")
    con.execute(f"CREATE VIEW acts   AS SELECT * FROM read_parquet('{ACTS}')")

    print("[features] base tokens table")
    base = con.sql(
        """
        SELECT
          token_id, deployer_address, deploy_time_unix,
          deployer_deposit_amount,
          deployer_wallet_balance_before,
          deployer_wallet_balance_after_sol,
          deployer_wallet_source_amount_sol,
          CASE WHEN deployer_wallet_source_is_cex THEN 1 ELSE 0 END AS is_cex,
          deployer_wallet_source_cex_name,
          CASE WHEN image_hash_sha256 IS NULL THEN 0 ELSE 1 END AS has_image,
          CASE WHEN description IS NULL OR description='' THEN 0 ELSE 1 END AS has_desc,
          CASE WHEN website IS NULL THEN 0 ELSE 1 END AS has_website,
          CASE WHEN twitter_handle IS NULL THEN 0 ELSE 1 END AS has_twitter,
          CASE WHEN telegram_link IS NULL THEN 0 ELSE 1 END AS has_telegram,
          LENGTH(name) AS name_len,
          LENGTH(ticker) AS ticker_len,
          LENGTH(description) AS desc_len,
          ath_market_cap_usd
        FROM tokens
        """
    ).pl()

    print("[features] deployer history")
    dh = deployer_history_features(con).with_columns(
        pl.col("deployer_prior_grad").cast(pl.Int64),
        pl.col("deployer_prior_hit20k").cast(pl.Int64),
        pl.col("deployer_prior_n").cast(pl.Int64),
        pl.col("deployer_seconds_since_last").cast(pl.Int64),
    )
    base = base.join(dh, on="token_id", how="left")

    print("[features] funder history")
    fh = funder_history_features(con).with_columns(
        pl.col("funder_prior_n").cast(pl.Int64),
        pl.col("funder_prior_hit20k").cast(pl.Int64),
        pl.col("funder_prior_grad").cast(pl.Int64),
    )
    base = base.join(fh, on="token_id", how="left")

    print("[features] regime / cross-section")
    base = base.join(regime_features(con), on="token_id", how="left")

    print("[features] clones")
    base = base.join(clone_features(con), on="token_id", how="left")

    print("[features] address vanity")
    base = base.join(address_features(con), on="token_id", how="left")

    print("[features] post-deploy aggregates")
    base = base.join(post_deploy_features(con), on="token_id", how="left")

    print("[features] labels")
    lab = label_features(con).with_columns(
        (pl.col("pmax_30m") / pl.col("p0")).alias("roi_30m"),
    ).with_columns(
        (pl.col("roi_30m") >= 2.0).cast(pl.Int8).alias("hit_2x"),
        (pl.col("roi_30m") >= 5.0).cast(pl.Int8).alias("hit_5x"),
    )
    base = base.join(lab, on="token_id", how="left")

    print("[features] time-of-day cyclical + market context")
    base = base.with_columns(
        ((pl.col("deploy_time_unix") % 86400) / 86400 * 2 * np.pi).alias("utc_phase"),
    ).with_columns(
        pl.col("utc_phase").sin().alias("utc_sin"),
        pl.col("utc_phase").cos().alias("utc_cos"),
        ((pl.col("deploy_time_unix") // 3600) % 24).alias("utc_hour"),
        (((pl.col("deploy_time_unix") // 86400) + 3) % 7).alias("utc_dow"),  # 1970-01-01 was Thu
        # Approx local hour buckets (April 2026: NY UTC-4, London UTC+1, Tokyo UTC+9)
        (((pl.col("deploy_time_unix") - 4 * 3600) // 3600) % 24).alias("ny_hour"),
        (((pl.col("deploy_time_unix") + 1 * 3600) // 3600) % 24).alias("ldn_hour"),
        (((pl.col("deploy_time_unix") + 9 * 3600) // 3600) % 24).alias("tokyo_hour"),
    )

    deploy_min = int(base["deploy_time_unix"].min())
    deploy_max = int(base["deploy_time_unix"].max())
    print(f"[features] fetching SOL/BTC OHLCV from {datetime.fromtimestamp(deploy_min, timezone.utc)} "
          f"to {datetime.fromtimestamp(deploy_max, timezone.utc)}")
    mkt = market_context(deploy_min, deploy_max).sort("ts_unix")

    base = base.sort("deploy_time_unix")
    base = base.join_asof(mkt, left_on="deploy_time_unix", right_on="ts_unix", strategy="backward")

    base.write_parquet(OUT / "features.parquet")
    summary = {
        "n_rows": base.height,
        "n_columns": base.width,
        "columns": base.columns,
        "label_n": int(base.drop_nulls(["hit_2x"]).height),
        "hit_2x_rate": float(base["hit_2x"].mean() or 0),
        "hit_5x_rate": float(base["hit_5x"].mean() or 0),
    }
    (OUT / "features_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
