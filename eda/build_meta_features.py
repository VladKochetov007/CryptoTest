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

LABEL_HORIZON_SEC = 1800


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

        # hi_resolved[i] = first j where times[j] >= times[i] - LABEL_HORIZON_SEC.
        # Peers at indices [lo, hi_resolved) have t_peer < t - 1800s, guaranteeing
        # their hit_2x label window closed before current token's deploy time.
        cum_full = np.concatenate([[0], np.cumsum(hits)])
        hi_resolved = np.searchsorted(times, times - LABEL_HORIZON_SEC, side="left")

        for wi, (w_name, w) in enumerate(zip(w_names, w_secs)):
            lo = np.searchsorted(times, times - w, side="left")
            prior_n = np.maximum(hi_resolved - lo, 0)
            prior_hits_arr = np.maximum(cum_full[hi_resolved] - cum_full[lo], 0)
            hit_rate = np.where(prior_n > 0, prior_hits_arr / np.maximum(prior_n, 1), np.nan)
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


def funder_graph_features(feat: pl.DataFrame, tokens: pl.DataFrame) -> pl.DataFrame:
    """Past-only graph features keyed on `deployer_wallet_source`.

    For each token at time t with funder F:
      funder_seconds_since_last  — t - max(prior_t) for this F (NaN if none).
      funder_unique_deployers_prior — distinct deployer addresses funded by F before t.
      funder_concentration_hhi      — Σ pᵢ² over deployer share distribution before t,
                                       where pᵢ = (#tokens by deployer i) / (total prior tokens).
      funder_avg_deposit_sol_prior  — mean deployer_wallet_source_amount_sol funded by F before t.
      funder_is_dust_funder         — 1 if amount<0.5 SOL AND prior_n>5.

    Time-respecting (strict <), tie-safe via stable sort + searchsorted side="left".
    """
    src = (
        feat.select("token_id", "deployer_address", "deploy_time_unix")
        .join(
            tokens.select("token_id", "deployer_wallet_source",
                          "deployer_wallet_source_amount_sol"),
            on="token_id", how="left",
        )
        .drop_nulls(["deployer_wallet_source"])
    )

    groups = src.group_by("deployer_wallet_source").agg(
        pl.col("token_id").alias("tids"),
        pl.col("deployer_address").alias("deps"),
        pl.col("deploy_time_unix").alias("times"),
        pl.col("deployer_wallet_source_amount_sol").alias("amts"),
    )

    rec_secs: dict[int, float] = {}
    rec_unique: dict[int, int] = {}
    rec_hhi: dict[int, float] = {}
    rec_amt: dict[int, float] = {}
    rec_dust: dict[int, int] = {}

    for row in groups.iter_rows(named=True):
        tids = np.asarray(row["tids"], dtype=np.int64)
        deps = np.asarray(row["deps"])
        times = np.asarray(row["times"], dtype=np.int64)
        amts = np.asarray(row["amts"], dtype=np.float64)

        order = np.argsort(times, kind="stable")
        tids = tids[order]; deps = deps[order]; times = times[order]; amts = amts[order]
        n_grp = len(tids)
        hi = np.searchsorted(times, times, side="left")  # excludes self + same-second peers

        # bookkeeping per deployer count
        from collections import defaultdict
        counter: dict[str, int] = defaultdict(int)
        cum_amt = 0.0
        cum_n_prior = 0
        unique_so_far = 0
        sumsq = 0.0  # Σ count_i^2 — used to derive HHI lazily
        last_seen_time: int | None = None
        # iterate in order. At step i, the "prior" snapshot is whatever we observed
        # in steps [0..hi[i] - 1]. To stay strict-time-precedes-self even for ties,
        # we snapshot AFTER catching up to hi[i] tokens — not after step i-1.
        snapshot_idx = 0
        for i in range(n_grp):
            target = hi[i]
            while snapshot_idx < target:
                d = deps[snapshot_idx]
                prev = counter[d]
                counter[d] = prev + 1
                if prev == 0:
                    unique_so_far += 1
                # Σ c_i^2 update from (prev)^2 to (prev+1)^2 = prev^2 + 2*prev + 1
                sumsq += 2 * prev + 1
                if not np.isnan(amts[snapshot_idx]):
                    cum_amt += float(amts[snapshot_idx])
                cum_n_prior += 1
                last_seen_time = int(times[snapshot_idx])
                snapshot_idx += 1
            tid = int(tids[i])
            rec_secs[tid] = float(times[i] - last_seen_time) if last_seen_time is not None else np.nan
            rec_unique[tid] = int(unique_so_far)
            if cum_n_prior > 0:
                hhi = sumsq / (cum_n_prior * cum_n_prior)
                rec_hhi[tid] = float(hhi)
                rec_amt[tid] = float(cum_amt / cum_n_prior)
            else:
                rec_hhi[tid] = float("nan")
                rec_amt[tid] = float("nan")
            this_amt = amts[i]
            rec_dust[tid] = int(
                (not np.isnan(this_amt)) and (this_amt < 0.5) and (cum_n_prior > 5)
            )

    all_tids = feat["token_id"].to_list()
    cols = [
        pl.Series("token_id", all_tids, dtype=pl.Int32),
        pl.Series("funder_seconds_since_last",
                  [rec_secs.get(t, float("nan")) for t in all_tids], dtype=pl.Float64),
        pl.Series("funder_unique_deployers_prior",
                  [rec_unique.get(t, 0) for t in all_tids], dtype=pl.Int32),
        pl.Series("funder_concentration_hhi",
                  [rec_hhi.get(t, float("nan")) for t in all_tids], dtype=pl.Float64),
        pl.Series("funder_avg_deposit_sol_prior",
                  [rec_amt.get(t, float("nan")) for t in all_tids], dtype=pl.Float64),
        pl.Series("funder_is_dust_funder",
                  [rec_dust.get(t, 0) for t in all_tids], dtype=pl.Int8),
    ]
    return pl.DataFrame(cols)


def twitter_handle_reuse_features(feat: pl.DataFrame, tokens: pl.DataFrame) -> pl.DataFrame:
    """Past-only sybil signal: distinct deployer count using same twitter_handle in last 7d."""
    src = (
        feat.select("token_id", "deployer_address", "deploy_time_unix")
        .join(tokens.select("token_id", "twitter_handle"), on="token_id", how="left")
        .drop_nulls(["twitter_handle"])
    )
    groups = src.group_by("twitter_handle").agg(
        pl.col("token_id").alias("tids"),
        pl.col("deployer_address").alias("deps"),
        pl.col("deploy_time_unix").alias("times"),
    )
    out: dict[int, int] = {}
    w = 604_800
    for row in groups.iter_rows(named=True):
        tids = np.asarray(row["tids"], dtype=np.int64)
        deps = np.asarray(row["deps"])
        times = np.asarray(row["times"], dtype=np.int64)
        order = np.argsort(times, kind="stable")
        tids = tids[order]; deps = deps[order]; times = times[order]
        hi = np.searchsorted(times, times, side="left")
        lo = np.searchsorted(times, times - w, side="left")
        for i, tid in enumerate(tids):
            window_deps = deps[lo[i]:hi[i]]
            out[int(tid)] = int(len(set(window_deps.tolist())))
    all_tids = feat["token_id"].to_list()
    return pl.DataFrame([
        pl.Series("token_id", all_tids, dtype=pl.Int32),
        pl.Series("handle_unique_deployers_7d",
                  [out.get(t, 0) for t in all_tids], dtype=pl.Int32),
    ])


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


def image_hash_rolling_count(feat: pl.DataFrame, tokens: pl.DataFrame) -> pl.DataFrame:
    """Count of prior tokens with the same SHA256 image hash (strictly past, 0 = no image or first use)."""
    tok_img = (
        feat.select("token_id", "deploy_time_unix")
        .join(tokens.select("token_id", "image_hash_sha256"), on="token_id", how="left")
    )
    all_tids = tok_img["token_id"].to_list()
    has_img = tok_img.filter(pl.col("image_hash_sha256").is_not_null())

    prior_counts: dict[int, int] = {}
    for row in has_img.group_by("image_hash_sha256").agg(
        pl.col("token_id").alias("tids"),
        pl.col("deploy_time_unix").alias("times"),
    ).iter_rows(named=True):
        times = np.array(row["times"], dtype=np.int64)
        tids = np.array(row["tids"], dtype=np.int32)
        order = np.argsort(times, kind="stable")
        for rank, tid in enumerate(tids[order]):
            prior_counts[int(tid)] = rank

    return pl.DataFrame({
        "token_id": all_tids,
        "image_hash_prior_count": pl.Series(
            [prior_counts.get(t, 0) for t in all_tids], dtype=pl.Int32
        ),
    })


def macro_derived_features(feat: pl.DataFrame) -> pl.DataFrame:
    """Stationary macro features derived from features.parquet columns.

    btc_ret_24h: 24h BTC return via join_asof on deploy timestamps (since many tokens
    share 5-min Binance bars, the token time series approximates the price series).
    sol_vol_ratio: short-term / long-term realized vol ratio (vol regime signal).
    sol_btc_ret_spread: SOL excess return vs BTC at 1h horizon.
    """
    macro = feat.select(
        "token_id", "deploy_time_unix",
        "btc_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "btc_ret_1h",
    ).sort("deploy_time_unix")

    price_series = (
        macro.select("deploy_time_unix", "btc_close")
        .unique("deploy_time_unix")
        .sort("deploy_time_unix")
        .rename({"deploy_time_unix": "ref_time", "btc_close": "btc_close_24h_ago"})
    )
    lookup = macro.with_columns(
        (pl.col("deploy_time_unix") - 86400).alias("ref_time")
    ).sort("ref_time")

    joined = lookup.join_asof(price_series, on="ref_time", strategy="nearest")

    return joined.with_columns(
        ((pl.col("btc_close") - pl.col("btc_close_24h_ago")) / pl.col("btc_close_24h_ago"))
        .alias("btc_ret_24h"),
        (pl.col("sol_vol_1h") / (pl.col("sol_vol_24h") + 1e-9)).alias("sol_vol_ratio"),
        (pl.col("sol_ret_1h") - pl.col("btc_ret_1h")).alias("sol_btc_ret_spread"),
    ).select("token_id", "btc_ret_24h", "sol_vol_ratio", "sol_btc_ret_spread")


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
                pl.col("n_96").shift(3).alias("n_prev"),
                pl.col("hits_96").shift(3).alias("hits_prev"),
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
        "btc_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "btc_ret_1h",
    )
    tokens = pl.read_parquet(TOKENS).select(
        "token_id", "name", "ticker", "description", "twitter_handle",
        "deployer_wallet_source", "deployer_wallet_source_amount_sol",
        "image_hash_sha256",
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
        {"24h": 86_400, "7d": 604_800}, "funder"
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

    t0 = time.time()
    print("[meta] funder graph features (unique deployers, hhi, dust, recency)")
    fund_graph = funder_graph_features(feat, tokens)
    print(f"       {fund_graph.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[meta] twitter handle deployer-reuse 7d")
    handle_reuse = twitter_handle_reuse_features(feat, tokens)
    print(f"       {handle_reuse.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[meta] image hash prior count (leak-free, strictly past)")
    img_count = image_hash_rolling_count(feat, tokens)
    print(f"       {img_count.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[meta] macro derived features (btc_ret_24h, sol_vol_ratio, sol_btc_ret_spread)")
    macro_feat = macro_derived_features(feat)
    print(f"       {macro_feat.shape} in {time.time()-t0:.1f}s")

    print("[meta] joining all features")
    base = feat.select("token_id")
    for df in [ms_dep, ms_fund, ms_handle_roll, handle_static, desc_feat,
                name_feat, meme_roll, fund_graph, handle_reuse, img_count, macro_feat]:
        base = base.join(df, on="token_id", how="left")

    out_path = OUT / "meta_features.parquet"
    base.write_parquet(out_path)
    print(f"[meta] saved {base.shape} -> {out_path}")


if __name__ == "__main__":
    main()
