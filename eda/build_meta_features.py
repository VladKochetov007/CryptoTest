"""Meta-features: multi-scale deployer/funder/handle windows + text features.

Approach for rolling window features: per-group cumsum + numpy searchsorted.
O(N log N) per window — ~7s total for 500k tokens × 4 windows.

All windows closed="left": only tokens with deploy_time < current token's time are counted.
hit_2x NULLs excluded from denominator of hit rate (no-trade tokens don't count as misses).

Output: eda/meta_features.parquet (token_id keyed, join with features.parquet)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "tokens.parquet"
FEATURES = ROOT / "eda" / "features.parquet"
OUT = ROOT / "eda"

MEME_KW = [
    "doge", "pepe", "trump", "ai", "bonk", "cat", "shib", "moon", "elon",
    "wojak", "chad", "frog", "based", "ape", "grok", "maga", "baby",
    "inu", "fire", "dragon", "bear", "bull", "pump", "sun", "bird", "dog",
    "meme", "wif", "sol", "crypto", "nft", "coin",
]

CELEB_HANDLES = {
    "elonmusk", "realdonaldtrump", "vitalikbuterin", "jack", "cz_binance",
    "michael_saylor", "aantonop", "balajis", "garyvee", "justinsuntron",
    "sbf_ftx", "punk6529", "cobie", "lookonchain", "solanafloor",
}

TEMPLATE_PHRASES = [
    "deployed using", "buy now", "to the moon", "100x", "safe", "gem",
    "rugproof", "rug proof", "fair launch", "no rug", "based team",
]


def compute_rolling_group_features(
    feat: pl.DataFrame,
    group_col: str,
    hit_col: str,
    windows: dict[str, int],
    prefix: str,
) -> pl.DataFrame:
    """For each token, compute prior count and prior hit rate at each window size.

    hit_col NULLs are excluded from count and hit rate (e.g., tokens with no price data).
    Uses numpy searchsorted per group — O(N log N), ~7s for 500k tokens.
    """
    sub = feat.select("token_id", group_col, "deploy_time_unix", hit_col).drop_nulls([group_col])
    has_hit = sub.filter(pl.col(hit_col).is_not_null())

    groups = has_hit.group_by(group_col).agg(
        pl.col("token_id").alias("tids"),
        pl.col("deploy_time_unix").alias("times"),
        pl.col(hit_col).fill_null(0).cast(pl.Int32).alias("hits"),
    )

    n_windows = len(windows)
    w_names = list(windows.keys())
    w_secs = list(windows.values())

    records_n: list[dict[int, int]] = [{} for _ in w_names]
    records_hr: list[dict[int, float]] = [{} for _ in w_names]

    for row in groups.iter_rows(named=True):
        times = np.array(row["times"], dtype=np.int64)
        hits = np.array(row["hits"], dtype=np.int32)
        tids = np.array(row["tids"], dtype=np.int32)

        order = np.argsort(times, kind="stable")
        times = times[order]
        hits = hits[order]
        tids = tids[order]

        # cum_hits[i] = sum of hits[0..i-1] (exclusive — hits before position i)
        cum_hits = np.concatenate([[0], np.cumsum(hits[:-1])]) if len(hits) > 1 else np.array([0])

        for wi, (w_name, w) in enumerate(zip(w_names, w_secs)):
            lo = np.searchsorted(times, times - w, side="left")
            prior_n = np.arange(len(times)) - lo
            prior_hits_arr = cum_hits - np.where(lo > 0, cum_hits[np.minimum(lo, len(cum_hits) - 1)], 0)
            hit_rate = np.where(prior_n > 0, prior_hits_arr / prior_n, np.nan)
            for i, tid in enumerate(tids):
                records_n[wi][int(tid)] = int(prior_n[i])
                records_hr[wi][int(tid)] = float(hit_rate[i])

    # build output DataFrame for ALL token_ids (including those with null group)
    all_tids = feat["token_id"].to_list()
    cols = [pl.Series("token_id", all_tids, dtype=pl.Int32)]
    for wi, w_name in enumerate(w_names):
        cols.append(pl.Series(
            f"{prefix}_prior_n_{w_name}",
            [records_n[wi].get(t, 0) for t in all_tids],
            dtype=pl.Int32,
        ))
        cols.append(pl.Series(
            f"{prefix}_hr_{w_name}",
            [records_hr[wi].get(t, float("nan")) for t in all_tids],
            dtype=pl.Float64,
        ))
    return pl.DataFrame(cols)


def twitter_handle_features(tokens: pl.DataFrame) -> pl.DataFrame:
    return (
        tokens.select("token_id", "ticker", "twitter_handle")
        .with_columns(
            pl.col("twitter_handle").str.len_chars().fill_null(0).alias("handle_len"),
            (
                pl.col("twitter_handle").str.count_matches(r"\d").fill_null(0)
                / pl.col("twitter_handle").str.len_chars().clip(lower_bound=1).fill_null(1)
            ).alias("handle_digit_ratio"),
            pl.col("twitter_handle").str.contains("_", literal=True)
              .fill_null(False).cast(pl.Int8).alias("handle_has_underscore"),
            pl.col("twitter_handle").str.contains(r"\d+$")
              .fill_null(False).cast(pl.Int8).alias("handle_ends_in_digits"),
            pl.col("twitter_handle").str.to_lowercase()
              .is_in(list(CELEB_HANDLES)).cast(pl.Int8).alias("handle_is_celeb"),
        )
        .with_columns(
            # starts_with/ends_with use literal matching (safe for special chars)
            (
                pl.col("twitter_handle").str.to_lowercase()
                  .str.starts_with(pl.col("ticker").str.to_lowercase())
                | pl.col("twitter_handle").str.to_lowercase()
                  .str.ends_with(pl.col("ticker").str.to_lowercase())
            ).fill_null(False).cast(pl.Int8).alias("handle_contains_ticker")
        )
        .select(
            "token_id", "handle_len", "handle_digit_ratio",
            "handle_has_underscore", "handle_ends_in_digits",
            "handle_is_celeb", "handle_contains_ticker",
        )
    )


def description_features(tokens: pl.DataFrame) -> pl.DataFrame:
    desc = tokens.select("token_id", pl.col("description").fill_null(""))
    template_score = sum(
        desc["description"].str.to_lowercase().str.contains(p).cast(pl.Int32)
        for p in TEMPLATE_PHRASES
    )
    return (
        desc.with_columns(
            pl.col("description").str.contains("http").cast(pl.Int8).alias("desc_has_url"),
            pl.col("description").str.to_lowercase().str.contains("deployed using")
              .cast(pl.Int8).alias("desc_has_deployed_template"),
            pl.col("description").str.count_matches("!").alias("desc_exclamation_count"),
            pl.Series("desc_template_score", template_score).alias("desc_template_score"),
            (pl.col("description").str.count_matches(r"\s+") + 1).alias("desc_word_count"),
            pl.col("description").str.count_matches(r"[A-HJ-NP-Z1-9]{30,}")
              .alias("desc_has_address"),
        )
        .select(
            "token_id", "desc_has_url", "desc_has_deployed_template",
            "desc_exclamation_count", "desc_template_score",
            "desc_word_count", "desc_has_address",
        )
    )


def name_text_features(tokens: pl.DataFrame) -> pl.DataFrame:
    kw_pattern = "|".join(MEME_KW)
    return (
        tokens.select("token_id", "name", "ticker")
        .with_columns(
            pl.col("name").str.to_lowercase().str.contains(kw_pattern)
              .cast(pl.Int8).alias("name_has_meme_kw"),
            pl.col("ticker").str.to_lowercase().str.contains(kw_pattern)
              .cast(pl.Int8).alias("ticker_has_meme_kw"),
            pl.col("name").str.count_matches(r"\s+").alias("name_word_count"),
            pl.col("name").str.count_matches(r"\d").alias("name_digit_count"),
            pl.col("ticker").str.count_matches(r"\d").alias("ticker_digit_count"),
            pl.col("ticker").str.count_matches(r"[^A-Za-z0-9]").alias("ticker_special_count"),
            pl.col("ticker").str.len_chars().is_between(4, 5).cast(pl.Int8).alias("ticker_len_4_5"),
        )
        .select(
            "token_id", "name_has_meme_kw", "ticker_has_meme_kw",
            "name_word_count", "name_digit_count", "ticker_digit_count",
            "ticker_special_count", "ticker_len_4_5",
        )
    )


def meme_kw_rolling_win_rate(feat: pl.DataFrame, name_feat: pl.DataFrame) -> pl.DataFrame:
    """Rolling hit_2x rate for tokens in same meme-kw category, last 24h.

    Bucket to 15-min intervals, rolling sum over past 96 buckets, shift by 1 bucket.
    O(N) — no per-token range join needed.
    """
    df = (
        feat.select("token_id", "deploy_time_unix", "hit_2x")
        .join(name_feat.select("token_id", "name_has_meme_kw"), on="token_id", how="left")
        .sort("deploy_time_unix")
    )
    results = []
    for kw_flag in [0, 1]:
        sub = df.filter(pl.col("name_has_meme_kw") == kw_flag)
        if sub.is_empty():
            continue
        per_bucket = (
            sub.with_columns((pl.col("deploy_time_unix") // 900).alias("bucket"))
            .group_by("bucket")
            .agg(pl.len().alias("n"), pl.col("hit_2x").cast(pl.Int32).sum().alias("hits"))
            .sort("bucket")
            .with_columns(
                pl.col("n").rolling_sum(window_size=96, min_periods=1).alias("n_96"),
                pl.col("hits").rolling_sum(window_size=96, min_periods=1).alias("hits_96"),
            )
            .with_columns(
                pl.col("n_96").shift(1).alias("n_prev"),
                pl.col("hits_96").shift(1).alias("hits_prev"),
            )
            .with_columns(
                (pl.col("hits_prev") / pl.col("n_prev").clip(lower_bound=1))
                  .alias("meme_kw_hr_24h")
            )
        )
        joined = (
            sub.with_columns((pl.col("deploy_time_unix") // 900).alias("bucket"))
            .join(per_bucket.select("bucket", "meme_kw_hr_24h"), on="bucket", how="left")
            .select("token_id", "meme_kw_hr_24h")
        )
        results.append(joined)

    if not results:
        return feat.select("token_id").with_columns(
            pl.lit(None).cast(pl.Float64).alias("meme_kw_hr_24h")
        )
    return pl.concat(results).sort("token_id")


def main():
    import time
    print("[meta] loading features + tokens")
    feat = pl.read_parquet(FEATURES).select(
        "token_id", "deployer_address", "deploy_time_unix", "hit_2x", "hit_5x",
    )
    tokens = pl.read_parquet(TOKENS).select(
        "token_id", "name", "ticker", "description", "twitter_handle",
        "deployer_wallet_source",
    )
    feat_full = feat.join(
        tokens.select("token_id", "deployer_wallet_source", "twitter_handle"),
        on="token_id", how="left"
    )
    n = feat.height
    print(f"  {n:,} tokens")

    t0 = time.time()
    print("[meta] multi-scale deployer features (4 windows)")
    ms_dep = compute_rolling_group_features(
        feat_full, "deployer_address", "hit_2x",
        {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}, "deployer"
    )
    print(f"       {ms_dep.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[meta] multi-scale funder features (2 windows)")
    ms_fund = compute_rolling_group_features(
        feat_full, "deployer_wallet_source", "hit_2x",
        {"24h": 86400, "7d": 604800}, "funder"
    )
    print(f"       {ms_fund.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[meta] twitter handle rolling features (1 window)")
    ms_handle_roll = compute_rolling_group_features(
        feat_full, "twitter_handle", "hit_2x",
        {"24h": 86400}, "handle"
    )
    print(f"       {ms_handle_roll.shape} in {time.time()-t0:.1f}s")

    print("[meta] text features (polars native)")
    handle_static = twitter_handle_features(tokens)
    desc_feat = description_features(tokens)
    name_feat = name_text_features(tokens)

    print("[meta] meme keyword rolling win rate")
    meme_roll = meme_kw_rolling_win_rate(feat, name_feat)
    print(f"       {meme_roll.shape}")

    print("[meta] joining all features")
    base = feat.select("token_id")
    for df in [ms_dep, ms_fund, ms_handle_roll, handle_static, desc_feat, name_feat, meme_roll]:
        base = base.join(df, on="token_id", how="left")

    out_path = OUT / "meta_features.parquet"
    base.write_parquet(out_path)
    print(f"[meta] saved {base.shape} -> {out_path}")


if __name__ == "__main__":
    main()
