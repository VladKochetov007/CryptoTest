"""Pump.fun pipeline: tokens.parquet + features.parquet -> meta features -> model -> backtest -> ranked CSV."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import time
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    import lightgbm as lgb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    ROOT = Path("/home/vlad/development/PumpTest")
    SLOTS = ROOT / "slot_features_60m.parquet"
    OUT_CSV = ROOT / "eda" / "scoring" / "scored_submission.csv"
    TRADE_LOG_CSV = ROOT / "eda" / "scoring" / "trade_log.csv"
    return (
        OUT_CSV,
        ROOT,
        SLOTS,
        TRADE_LOG_CSV,
        defaultdict,
        lgb,
        mo,
        np,
        pl,
        plt,
        roc_auc_score,
        time,
    )


@app.cell
def _():
    # --- pipeline constants ---
    LABEL_HORIZON_SEC = 1800   # hit_2x label resolves at t_peer + 1800s
    LATENCY_SEC = 1            # first bot-accessible slot (slot 0 = deployer devbuy)
    POSITION_SOL = 0.1         # size per trade in SOL
    ARM_MULT = 1.5             # trailing stop arms when price hits 1.5x entry
    TRAIL_FRAC = 0.70          # exit when price drops to 70% of running max
    SL_FRAC = 0.40             # hard stop: exit at 40% of entry price
    CAT_COL = "deployer_wallet_source_cex_name"

    # pump.fun bonding-curve invariant: K = v_sol_init * v_token_init
    # v_sol_init = 30 SOL, v_token_init = 1.073e9 tokens (human units)
    K_INVARIANT = 30.0 * 1.073e9
    TAKER_FEE = 0.01
    return (
        ARM_MULT,
        CAT_COL,
        K_INVARIANT,
        LABEL_HORIZON_SEC,
        LATENCY_SEC,
        POSITION_SOL,
        SL_FRAC,
        TAKER_FEE,
        TRAIL_FRAC,
    )


@app.cell
def _(K_INVARIANT, TAKER_FEE):
    # --- inlined AMM math (from eda/amm.py) ---
    def _reserves(price):
        v_sol = (K_INVARIANT * price) ** 0.5
        v_tok = (K_INVARIANT / price) ** 0.5
        return v_sol, v_tok

    def amm_buy_tokens(price, sol_in):
        """SOL in -> tokens received. Returns (tokens, sol_spent_per_token)."""
        v_sol, v_tok = _reserves(price)
        sol_curve = sol_in * (1.0 - TAKER_FEE)
        new_v_tok = K_INVARIANT / (v_sol + sol_curve)
        tokens_out = v_tok - new_v_tok
        if tokens_out <= 0:
            return 0.0, float("inf")
        return tokens_out, sol_in / tokens_out

    def amm_sell_sol(price, tokens_in):
        """Tokens in -> SOL received (after fee)."""
        if tokens_in <= 0:
            return 0.0
        v_sol, v_tok = _reserves(price)
        new_v_sol = K_INVARIANT / (v_tok + tokens_in)
        sol_pre_fee = v_sol - new_v_sol
        return max(sol_pre_fee * (1.0 - TAKER_FEE), 0.0)

    def round_trip_net_roi(entry_price, exit_price, sol_in):
        """Net ROI after buy + sell fills. Negative = loss."""
        tokens, _ = amm_buy_tokens(entry_price, sol_in)
        if tokens <= 0:
            return -1.0
        sol_out = amm_sell_sol(exit_price, tokens)
        return sol_out / sol_in - 1.0

    # quick sanity: 2x move on 0.1 SOL should give ~96% net ROI (fee drag ~4%)
    _tok, _ = amm_buy_tokens(2.8e-8, 0.1)
    _sol_out = amm_sell_sol(2 * 2.8e-8, _tok)
    assert abs(_sol_out / 0.1 - 2.0) < 0.1, f"AMM sanity failed: net={_sol_out/0.1:.3f}"
    return amm_buy_tokens, amm_sell_sol


@app.cell
def _(LABEL_HORIZON_SEC, defaultdict, np, pl):
    # --- inlined feature functions (from eda/build_meta_features.py) ---

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

    def compute_rolling_group_features(feat, group_col, hit_col, windows, prefix):
        sub = feat.select("token_id", group_col, "deploy_time_unix", hit_col).drop_nulls([group_col])
        has_hit = sub.filter(pl.col(hit_col).is_not_null())
        groups = has_hit.group_by(group_col).agg(
            pl.col("token_id").alias("tids"),
            pl.col("deploy_time_unix").alias("times"),
            pl.col(hit_col).fill_null(0).cast(pl.Int32).alias("hits"),
        )
        w_names = list(windows.keys())
        w_secs = list(windows.values())
        records_n = [{} for _ in w_names]
        records_hr = [{} for _ in w_names]
        for row in groups.iter_rows(named=True):
            times = np.array(row["times"], dtype=np.int64)
            hits = np.array(row["hits"], dtype=np.int32)
            tids = np.array(row["tids"], dtype=np.int32)
            order = np.argsort(times, kind="stable")
            times = times[order]; hits = hits[order]; tids = tids[order]
            cum_full = np.concatenate([[0], np.cumsum(hits)])
            # hi_resolved[i]: first j where t[j] >= t[i] - 1800s
            # peers in [lo, hi_resolved) have closed label windows
            hi_resolved = np.searchsorted(times, times - LABEL_HORIZON_SEC, side="left")
            for wi, (w_name, w) in enumerate(zip(w_names, w_secs)):
                lo = np.searchsorted(times, times - w, side="left")
                prior_n = np.maximum(hi_resolved - lo, 0)
                prior_hits_arr = np.maximum(cum_full[hi_resolved] - cum_full[lo], 0)
                hit_rate = np.where(prior_n > 0, prior_hits_arr / np.maximum(prior_n, 1), np.nan)
                for i, tid in enumerate(tids):
                    records_n[wi][int(tid)] = int(prior_n[i])
                    records_hr[wi][int(tid)] = float(hit_rate[i])
        all_tids = feat["token_id"].to_list()
        cols = [pl.Series("token_id", all_tids, dtype=pl.Int32)]
        for wi, w_name in enumerate(w_names):
            cols.append(pl.Series(f"{prefix}_prior_n_{w_name}",
                [records_n[wi].get(t, 0) for t in all_tids], dtype=pl.Int32))
            cols.append(pl.Series(f"{prefix}_hr_{w_name}",
                [records_hr[wi].get(t, float("nan")) for t in all_tids], dtype=pl.Float64))
        return pl.DataFrame(cols)

    def funder_graph_features(feat, tokens):
        src = (
            feat.select("token_id", "deployer_address", "deploy_time_unix")
            .join(tokens.select("token_id", "deployer_wallet_source",
                                "deployer_wallet_source_amount_sol"), on="token_id", how="left")
            .drop_nulls(["deployer_wallet_source"])
        )
        groups = src.group_by("deployer_wallet_source").agg(
            pl.col("token_id").alias("tids"),
            pl.col("deployer_address").alias("deps"),
            pl.col("deploy_time_unix").alias("times"),
            pl.col("deployer_wallet_source_amount_sol").alias("amts"),
        )
        rec_secs = {}; rec_unique = {}; rec_hhi = {}; rec_amt = {}; rec_dust = {}
        for row in groups.iter_rows(named=True):
            tids = np.asarray(row["tids"], dtype=np.int64)
            deps = np.asarray(row["deps"])
            times = np.asarray(row["times"], dtype=np.int64)
            amts = np.asarray(row["amts"], dtype=np.float64)
            order = np.argsort(times, kind="stable")
            tids = tids[order]; deps = deps[order]; times = times[order]; amts = amts[order]
            hi = np.searchsorted(times, times, side="left")
            counter = defaultdict(int)
            cum_amt = 0.0; cum_n_prior = 0; unique_so_far = 0; sumsq = 0.0
            last_seen_time = None; snapshot_idx = 0
            for i in range(len(tids)):
                target = hi[i]
                while snapshot_idx < target:
                    d = deps[snapshot_idx]
                    prev = counter[d]
                    counter[d] = prev + 1
                    if prev == 0:
                        unique_so_far += 1
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
                    rec_hhi[tid] = float(sumsq / (cum_n_prior * cum_n_prior))
                    rec_amt[tid] = float(cum_amt / cum_n_prior)
                else:
                    rec_hhi[tid] = float("nan"); rec_amt[tid] = float("nan")
                this_amt = amts[i]
                rec_dust[tid] = int((not np.isnan(this_amt)) and (this_amt < 0.5) and (cum_n_prior > 5))
        all_tids = feat["token_id"].to_list()
        return pl.DataFrame([
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
        ])

    def twitter_handle_reuse_features(feat, tokens):
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
        out = {}
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
                out[int(tid)] = int(len(set(deps[lo[i]:hi[i]].tolist())))
        all_tids = feat["token_id"].to_list()
        return pl.DataFrame([
            pl.Series("token_id", all_tids, dtype=pl.Int32),
            pl.Series("handle_unique_deployers_7d", [out.get(t, 0) for t in all_tids], dtype=pl.Int32),
        ])

    def twitter_handle_features(tokens):
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
                (
                    pl.col("twitter_handle").str.to_lowercase()
                      .str.starts_with(pl.col("ticker").str.to_lowercase())
                    | pl.col("twitter_handle").str.to_lowercase()
                      .str.ends_with(pl.col("ticker").str.to_lowercase())
                ).fill_null(False).cast(pl.Int8).alias("handle_contains_ticker")
            )
            .select("token_id", "handle_len", "handle_digit_ratio",
                    "handle_has_underscore", "handle_ends_in_digits",
                    "handle_is_celeb", "handle_contains_ticker")
        )

    def description_features(tokens):
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
            .select("token_id", "desc_has_url", "desc_has_deployed_template",
                    "desc_exclamation_count", "desc_template_score",
                    "desc_word_count", "desc_has_address")
        )

    def name_text_features(tokens):
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
            .select("token_id", "name_has_meme_kw", "ticker_has_meme_kw",
                    "name_word_count", "name_digit_count", "ticker_digit_count",
                    "ticker_special_count", "ticker_len_4_5")
        )

    def image_hash_rolling_count(feat, tokens):
        tok_img = (
            feat.select("token_id", "deploy_time_unix")
            .join(tokens.select("token_id", "image_hash_sha256"), on="token_id", how="left")
        )
        all_tids = tok_img["token_id"].to_list()
        has_img = tok_img.filter(pl.col("image_hash_sha256").is_not_null())
        prior_counts = {}
        for row in has_img.group_by("image_hash_sha256").agg(
            pl.col("token_id").alias("tids"),
            pl.col("deploy_time_unix").alias("times"),
        ).iter_rows(named=True):
            times = np.array(row["times"], dtype=np.int64)
            tids = np.array(row["tids"], dtype=np.int32)
            order = np.argsort(times, kind="stable")
            times_s = times[order]
            tids_s = tids[order]
            # use searchsorted so same-timestamp tokens all see the same prior count
            for i, tid in enumerate(tids_s):
                prior_counts[int(tid)] = int(np.searchsorted(times_s, times_s[i], side="left"))
        return pl.DataFrame({
            "token_id": all_tids,
            "image_hash_prior_count": pl.Series(
                [prior_counts.get(t, 0) for t in all_tids], dtype=pl.Int32),
        })

    def macro_derived_features(feat):
        macro = feat.select(
            "token_id", "deploy_time_unix",
            "btc_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "btc_ret_1h",
        ).sort("deploy_time_unix")
        price_series = (
            macro.select("deploy_time_unix", "btc_close")
            .unique("deploy_time_unix").sort("deploy_time_unix")
            .rename({"deploy_time_unix": "ref_time", "btc_close": "btc_close_24h_ago"})
        )
        lookup = macro.with_columns(
            (pl.col("deploy_time_unix") - 86400).alias("ref_time")
        ).sort("ref_time")
        joined = lookup.join_asof(price_series, on="ref_time", strategy="backward")
        return joined.with_columns(
            ((pl.col("btc_close") - pl.col("btc_close_24h_ago")) / pl.col("btc_close_24h_ago"))
            .alias("btc_ret_24h"),
            (pl.col("sol_vol_1h") / (pl.col("sol_vol_24h") + 1e-9)).alias("sol_vol_ratio"),
            (pl.col("sol_ret_1h") - pl.col("btc_ret_1h")).alias("sol_btc_ret_spread"),
        ).select("token_id", "btc_ret_24h", "sol_vol_ratio", "sol_btc_ret_spread")

    def meme_kw_rolling_win_rate(feat, name_feat):
        # shift(3) = 45 min lag so every counted peer's 30-min label window is closed
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
                pl.lit(None).cast(pl.Float64).alias("meme_kw_hr_24h"))
        return pl.concat(results).sort("token_id")

    return (
        compute_rolling_group_features,
        description_features,
        funder_graph_features,
        image_hash_rolling_count,
        macro_derived_features,
        meme_kw_rolling_win_rate,
        name_text_features,
        twitter_handle_features,
        twitter_handle_reuse_features,
    )


@app.cell
def _(CAT_COL, np, pl):
    # --- inlined model utils (from eda/meta_train.py) ---

    def walkforward_splits(times, n_folds=5, embargo_sec=3600):
        n = len(times)
        edges = np.linspace(0, n, n_folds + 2, dtype=int)
        splits = []
        for k in range(n_folds):
            val_start = edges[k + 1]
            val_end = edges[k + 2]
            if val_end - val_start <= 0:
                continue
            cutoff = times[val_start] - embargo_sec
            train_end = int(np.searchsorted(times, cutoff, side="left"))
            if train_end <= 0:
                continue
            splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
        return splits

    def prep(df, cols, target):
        sub = df.select(cols + [target])
        if CAT_COL in cols:
            sub = sub.with_columns(
                pl.col(CAT_COL).fill_null("__missing__").cast(pl.Categorical)
            )
        numeric = [c for c in cols if c != CAT_COL]
        sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
        pdf = sub.to_pandas()
        X = pdf[cols]
        y = pdf[target].astype(int).values
        cat_idx = [cols.index(CAT_COL)] if CAT_COL in cols else []
        return X, y, cat_idx

    return prep, walkforward_splits


@app.cell
def _(ROOT, pl, time):
    _t0 = time.time()
    feat_raw = pl.read_parquet(ROOT / "eda" / "features.parquet")
    tokens = pl.read_parquet(ROOT / "tokens.parquet").select(
        "token_id", "name", "ticker", "description", "twitter_handle",
        "deployer_wallet_source", "deployer_wallet_source_amount_sol",
        "image_hash_sha256",
    )
    feat = feat_raw.select(
        "token_id", "deployer_address", "deploy_time_unix", "hit_2x", "hit_5x",
        "btc_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "btc_ret_1h",
    )
    feat_full = feat.join(
        tokens.select("token_id", "deployer_wallet_source", "twitter_handle"),
        on="token_id", how="left",
    )
    _load_sec = time.time() - _t0
    return feat, feat_full, feat_raw, tokens


@app.cell
def _(
    compute_rolling_group_features,
    description_features,
    feat,
    feat_full,
    funder_graph_features,
    image_hash_rolling_count,
    macro_derived_features,
    meme_kw_rolling_win_rate,
    name_text_features,
    time,
    tokens,
    twitter_handle_features,
    twitter_handle_reuse_features,
):
    _t0 = time.time()
    _ms_dep = compute_rolling_group_features(
        feat_full, "deployer_address", "hit_2x",
        {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}, "deployer",
    )
    _ms_fund = compute_rolling_group_features(
        feat_full, "deployer_wallet_source", "hit_2x",
        {"24h": 86_400, "7d": 604_800}, "funder",
    )
    _ms_handle_roll = compute_rolling_group_features(
        feat_full, "twitter_handle", "hit_2x",
        {"24h": 86400}, "handle",
    )
    _handle_static = twitter_handle_features(tokens)
    _desc_feat = description_features(tokens)
    _name_feat = name_text_features(tokens)
    _meme_roll = meme_kw_rolling_win_rate(feat, _name_feat)
    _fund_graph = funder_graph_features(feat, tokens)
    _handle_reuse = twitter_handle_reuse_features(feat, tokens)
    _img_count = image_hash_rolling_count(feat, tokens)
    _macro_feat = macro_derived_features(feat)

    _base = feat.select("token_id")
    for _df in [_ms_dep, _ms_fund, _ms_handle_roll, _handle_static, _desc_feat,
                _name_feat, _meme_roll, _fund_graph, _handle_reuse, _img_count, _macro_feat]:
        _base = _base.join(_df, on="token_id", how="left")

    meta_features_df = _base
    meta_elapsed = time.time() - _t0
    return meta_elapsed, meta_features_df


@app.cell
def _(feat_raw, meta_elapsed, meta_features_df, mo):
    _feat_base = feat_raw.drop_nulls("hit_2x").sort("deploy_time_unix")
    df = _feat_base.join(meta_features_df, on="token_id", how="left")
    mo.md(f"**Data ready** — {df.height:,} labeled tokens, meta features computed in {meta_elapsed:.1f}s")
    return (df,)


@app.cell
def _():
    # FORBIDDEN: ATH/hit20k resolve over unbounded future horizon.
    # raw price levels (btc_close, sol_close) are non-stationary regime leaks.
    _FORBIDDEN = {
        "deployer_prior_grad", "deployer_prior_hit20k",
        "funder_prior_hit20k", "funder_prior_grad",
        "hit20k_rate_prev_60m", "image_hash_seen_total",
        "btc_close", "sol_close",
    }

    BASE_FEATURES = [
        "deployer_deposit_amount",
        "deployer_wallet_balance_before",
        "deployer_wallet_balance_after_sol",
        "deployer_wallet_source_amount_sol",
        "is_cex",
        "deployer_wallet_source_cex_name",
        "has_image", "has_website", "has_telegram",
        "name_len", "ticker_len", "desc_len",
        "deployer_prior_n", "deployer_seconds_since_last",
        "funder_prior_n",
        "deploys_prev_15m", "deploys_prev_60m",
        "same_ticker_today_prev", "same_name_prev_hour",
        "mint_suffix_pump", "deployer_suffix_pump",
        "name_alpha_chars", "name_upper_chars",
        "utc_sin", "utc_cos", "utc_hour", "utc_dow", "ny_hour", "ldn_hour", "tokyo_hour",
        "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "sol_ret_24h",
        "btc_vol_1h", "btc_ret_1h",
    ]
    META_FEATURES = [
        "deployer_prior_n_1h", "deployer_prior_n_6h", "deployer_prior_n_24h", "deployer_prior_n_7d",
        "deployer_hr_1h", "deployer_hr_24h", "deployer_hr_7d",
        "funder_prior_n_24h", "funder_hr_24h",
        "funder_prior_n_7d", "funder_hr_7d",
        "funder_seconds_since_last", "funder_unique_deployers_prior",
        "funder_concentration_hhi", "funder_avg_deposit_sol_prior", "funder_is_dust_funder",
        "handle_unique_deployers_7d",
        "handle_len", "handle_digit_ratio", "handle_has_underscore",
        "handle_ends_in_digits", "handle_is_celeb", "handle_contains_ticker",
        "handle_prior_n_24h", "handle_hr_24h",
        "desc_has_url", "desc_has_deployed_template", "desc_exclamation_count",
        "desc_template_score", "desc_word_count", "desc_has_address",
        "name_has_meme_kw", "ticker_has_meme_kw",
        "name_word_count", "name_digit_count", "ticker_digit_count",
        "ticker_special_count", "ticker_len_4_5",
        "meme_kw_hr_24h",
        "image_hash_prior_count",
        "btc_ret_24h", "sol_vol_ratio", "sol_btc_ret_spread",
    ]

    assert not (_FORBIDDEN & set(BASE_FEATURES)), f"forbidden in BASE: {_FORBIDDEN & set(BASE_FEATURES)}"
    assert not (_FORBIDDEN & set(META_FEATURES)), f"forbidden in META: {_FORBIDDEN & set(META_FEATURES)}"
    return BASE_FEATURES, META_FEATURES


@app.cell
def _(
    BASE_FEATURES,
    META_FEATURES,
    df,
    lgb,
    np,
    prep,
    roc_auc_score,
    walkforward_splits,
):
    _all_cols = BASE_FEATURES + META_FEATURES
    model_cols = [c for c in _all_cols if c in df.columns]

    _times = df["deploy_time_unix"].to_numpy()
    _splits = walkforward_splits(_times, n_folds=5, embargo_sec=3600)
    _n = df.height
    oof = np.full(_n, np.nan)
    fold_aucs = []
    _booster = None

    for _k, (_tr_idx, _va_idx) in enumerate(_splits):
        _tr, _va = df[_tr_idx], df[_va_idx]
        _X_tr, _y_tr, _cat_idx = prep(_tr, model_cols, "hit_2x")
        _X_va, _y_va, _ = prep(_va, model_cols, "hit_2x")
        _base_rate = float(_y_tr.mean())
        _ds_tr = lgb.Dataset(_X_tr, label=_y_tr, categorical_feature=_cat_idx, free_raw_data=False)
        _ds_va = lgb.Dataset(_X_va, label=_y_va, free_raw_data=False)
        _params = dict(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            min_data_in_leaf=200,
            scale_pos_weight=(1 - _base_rate) / max(_base_rate, 1e-6),
            verbose=-1, n_jobs=-1,
        )
        _booster = lgb.train(
            _params, _ds_tr, num_boost_round=600,
            valid_sets=[_ds_va],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        _pred = _booster.predict(_X_va)
        oof[_va_idx] = _pred
        fold_aucs.append(float(roc_auc_score(_y_va, _pred)))

    booster = _booster
    return booster, fold_aucs, model_cols, oof


@app.cell
def _(df, fold_aucs, mo, np, oof, pl, roc_auc_score):
    _mask = ~np.isnan(oof)
    oof_auc = float(roc_auc_score(df["hit_2x"].to_numpy()[_mask].astype(int), oof[_mask]))

    _fold_tbl = pl.DataFrame({
        "fold": list(range(len(fold_aucs))),
        "auc": [round(a, 4) for a in fold_aucs],
    })
    mo.hstack([
        mo.stat(label="OOF AUC", value=f"{oof_auc:.4f}"),
        mo.stat(label="n_tokens", value=f"{_mask.sum():,}"),
        mo.stat(label="base_rate", value=f"{df['hit_2x'].mean():.3f}"),
        mo.vstack([mo.md("**Fold AUCs**"), _fold_tbl]),
    ])
    return


@app.cell
def _(booster, model_cols, np, plt):
    _gain = booster.feature_importance(importance_type="gain")
    _idx = np.argsort(_gain)[::-1][:30]
    _names = [model_cols[i] for i in _idx]
    _vals = _gain[_idx] / max(_gain.sum(), 1.0)

    fig_imp, _ax = plt.subplots(figsize=(10, 10))
    _ax.barh(range(len(_names)), _vals[::-1])
    _ax.set_yticks(range(len(_names)))
    _ax.set_yticklabels(_names[::-1], fontsize=8.5)
    _ax.set_xlabel("gain share")
    _ax.set_title("Top 30 features — LightGBM gain")
    plt.tight_layout()
    fig_imp
    return


@app.cell
def _(df, np, oof):
    _mask = ~np.isnan(oof)
    _thresh = float(np.quantile(oof[_mask], 0.9))

    _valid_ids = df["token_id"].to_numpy()[_mask]
    _valid_oof = oof[_mask]

    model_ids = _valid_ids[_valid_oof >= _thresh].tolist()
    model_oof_scores = dict(zip(
        _valid_ids[_valid_oof >= _thresh].tolist(),
        _valid_oof[_valid_oof >= _thresh].tolist(),
    ))

    rng = np.random.default_rng(42)
    random_ids = rng.choice(_valid_ids, size=len(model_ids), replace=False).tolist()

    hit2x_lookup = dict(zip(df["token_id"].to_list(), df["hit_2x"].to_list()))
    return hit2x_lookup, model_ids, model_oof_scores, random_ids


@app.cell
def _(SLOTS, model_ids, pl, random_ids):
    _all_ids = pl.Series("token_id", list(set(model_ids) | set(random_ids)), dtype=pl.Int32)
    _raw = (
        pl.scan_parquet(SLOTS)
        .filter(pl.col("token_id").is_in(_all_ids) & (pl.col("price_sol_per_token") > 0))
        .select("token_id", "seconds_since_deploy", "price_sol_per_token")
        .sort(["token_id", "seconds_since_deploy"])
        .collect()
    )
    panels = {t[0]: g.sort("seconds_since_deploy") for t, g in _raw.group_by("token_id")}
    return (panels,)


@app.cell
def _(
    ARM_MULT,
    LATENCY_SEC,
    POSITION_SOL,
    SL_FRAC,
    TRAIL_FRAC,
    amm_buy_tokens,
    amm_sell_sol,
    hit2x_lookup,
    model_ids,
    model_oof_scores,
    np,
    panels,
    pl,
    random_ids,
):
    def sim_trailing(tid, secs, prices, sol_in, latency_sec,
                     arm_mult, trail_frac, sl_frac):
        idx0 = int(np.searchsorted(secs, latency_sec, side="left"))
        if idx0 >= len(prices):
            return None

        slot0_price = prices[0]
        entry_price = prices[idx0]
        entry_sec = int(secs[idx0])
        tokens_held, _ = amm_buy_tokens(entry_price, sol_in)
        if tokens_held <= 0:
            return None

        peak_price = entry_price
        armed = False

        for i in range(idx0 + 1, len(prices)):
            p = prices[i]
            if p > peak_price:
                peak_price = p
            if not armed and p >= arm_mult * entry_price:
                armed = True
            if armed and p <= trail_frac * peak_price:
                sol_out = amm_sell_sol(p, tokens_held)
                return {
                    "exit_price": p, "exit_sec": int(secs[i]),
                    "reason": "trail", "sol_out": sol_out,
                    "peak_price": peak_price, "entry_price": entry_price,
                    "entry_sec": entry_sec, "slot0_price": slot0_price,
                    "tokens_held": tokens_held,
                }
            if p <= sl_frac * entry_price:
                sol_out = amm_sell_sol(p, tokens_held)
                return {
                    "exit_price": p, "exit_sec": int(secs[i]),
                    "reason": "sl_hard", "sol_out": sol_out,
                    "peak_price": peak_price, "entry_price": entry_price,
                    "entry_sec": entry_sec, "slot0_price": slot0_price,
                    "tokens_held": tokens_held,
                }

        sol_out = amm_sell_sol(prices[-1], tokens_held)
        return {
            "exit_price": prices[-1], "exit_sec": int(secs[-1]),
            "reason": "timeout", "sol_out": sol_out,
            "peak_price": peak_price, "entry_price": entry_price,
            "entry_sec": entry_sec, "slot0_price": slot0_price,
            "tokens_held": tokens_held,
        }

    def run_backtest(ids, arm, oof_map):
        rows = []
        for tid in ids:
            panel = panels.get(tid)
            if panel is None:
                continue
            sub = panel.filter(pl.col("seconds_since_deploy") <= 1800)
            if sub.height == 0:
                continue
            secs = sub["seconds_since_deploy"].to_numpy()
            prices = sub["price_sol_per_token"].to_numpy()
            res = sim_trailing(
                tid, secs, prices, POSITION_SOL, LATENCY_SEC,
                ARM_MULT, TRAIL_FRAC, SL_FRAC,
            )
            if res is None:
                continue
            entry_price = res["entry_price"]
            exit_price = res["exit_price"]
            sol_out = res["sol_out"]
            gross_roi = exit_price / entry_price - 1.0
            net_roi = sol_out / POSITION_SOL - 1.0
            rows.append({
                "token_id": tid,
                "arm": arm,
                "oof_score": float(oof_map.get(tid, float("nan"))),
                "hit_2x_actual": int(hit2x_lookup.get(tid, 0) or 0),
                "slot0_price": res["slot0_price"],
                "entry_price": entry_price,
                "entry_sec": res["entry_sec"],
                "entry_drift": (entry_price / res["slot0_price"] - 1.0) if res["slot0_price"] > 0 else float("nan"),
                "peak_mult": res["peak_price"] / entry_price,
                "exit_price": exit_price,
                "exit_sec": res["exit_sec"],
                "hold_sec": res["exit_sec"] - res["entry_sec"],
                "exit_reason": res["reason"],
                "gross_roi": gross_roi,
                "net_roi": net_roi,
                "cost_drag": gross_roi - net_roi,
            })
        return rows

    _dummy_oof = {tid: float("nan") for tid in random_ids}
    trades_model = run_backtest(model_ids, "model", model_oof_scores)
    trades_random = run_backtest(random_ids, "random", _dummy_oof)

    trade_log = pl.DataFrame(trades_model + trades_random)
    return trade_log, trades_model, trades_random


@app.cell
def _(POSITION_SOL, mo, np, pl, plt, trade_log, trades_model, trades_random):
    def _stats(rows, label):
        if not rows:
            return {"universe": label, "n": 0}
        gross = np.array([r["gross_roi"] for r in rows])
        net = np.array([r["net_roi"] for r in rows])
        pos_net = net[net > 0]
        top10_share = float(np.sort(pos_net)[::-1][:10].sum() / pos_net.sum()) if len(pos_net) else float("nan")
        hit2x = np.array([r["hit_2x_actual"] for r in rows])
        return {
            "universe": label,
            "n": len(rows),
            "hit_2x_rate": f"{hit2x.mean():.3f}",
            "gross_mean_roi": f"{gross.mean():.4f}",
            "net_mean_roi": f"{net.mean():.4f}",
            "gross_median_roi": f"{float(np.median(gross)):.4f}",
            "net_median_roi": f"{float(np.median(net)):.4f}",
            "net_win_rate": f"{(net > 0).mean():.3f}",
            "net_pnl_sol": f"{net.sum() * POSITION_SOL:.3f}",
            "top10_pos_pnl_share": f"{top10_share:.3f}",
            "mean_cost_drag": f"{(gross - net).mean():.4f}",
            "mean_peak_mult": f"{np.mean([r['peak_mult'] for r in rows]):.3f}",
        }

    _sm = sorted(trades_model, key=lambda r: r["exit_sec"])
    _sr = sorted(trades_random, key=lambda r: r["exit_sec"])
    _cum_net_m = np.cumsum([r["net_roi"] * POSITION_SOL for r in _sm])
    _cum_net_r = np.cumsum([r["net_roi"] * POSITION_SOL for r in _sr])
    _cum_gross_m = np.cumsum([r["gross_roi"] * POSITION_SOL for r in _sm])

    from collections import Counter as _Counter
    _reasons_m = _Counter(r["exit_reason"] for r in trades_model)
    _reasons_r = _Counter(r["exit_reason"] for r in trades_random)

    fig_diag, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    ax1.plot(_cum_net_m, lw=1.8, label=f"model net (n={len(trades_model)})", color="#2563eb")
    ax1.plot(_cum_net_r, lw=1.2, label=f"random net (n={len(trades_random)})", color="#dc2626", alpha=0.7)
    ax1.axhline(0, color="grey", lw=0.7, ls="--")
    ax1.set_title("Cumulative net PnL (SOL)")
    ax1.set_xlabel("trade # by exit time"); ax1.legend()

    ax2.plot(_cum_gross_m, lw=1.8, color="#1d4ed8", label="model gross")
    ax2.axhline(0, color="grey", lw=0.7, ls="--")
    ax2.set_title("Model gross PnL (no costs)")
    ax2.set_xlabel("trade # by exit time"); ax2.legend()

    _all_r = sorted(set(_reasons_m) | set(_reasons_r))
    _x = np.arange(len(_all_r)); _w = 0.35
    ax3.bar(_x - _w/2, [_reasons_m.get(r, 0) for r in _all_r], _w, label="model", color="#2563eb")
    ax3.bar(_x + _w/2, [_reasons_r.get(r, 0) for r in _all_r], _w, label="random", color="#dc2626", alpha=0.8)
    ax3.set_xticks(_x); ax3.set_xticklabels(_all_r)
    ax3.set_title("Exit reason distribution"); ax3.legend()

    _cost_drag = trade_log.filter(pl.col("arm") == "model")["cost_drag"].to_numpy()
    ax4.hist(_cost_drag, bins=60, color="#7c3aed", alpha=0.8)
    ax4.set_title("Cost drag per trade (gross - net ROI)")
    ax4.set_xlabel("ROI drag")

    _peak = trade_log.filter(pl.col("arm") == "model")["peak_mult"].to_numpy()
    ax5.hist(_peak, bins=60, color="#059669", alpha=0.8)
    ax5.axvline(1.5, color="red", lw=1.2, ls="--", label="arm threshold 1.5x")
    ax5.set_title("Peak multiplier distribution (model arm)")
    ax5.set_xlabel("peak / entry"); ax5.legend()

    _drift = trade_log.filter(pl.col("arm") == "model")["entry_drift"].drop_nulls().to_numpy()
    ax6.hist(_drift, bins=60, color="#d97706", alpha=0.8)
    ax6.set_title("Entry drift: (entry_price / slot0_price) - 1")
    ax6.set_xlabel("drift (fraction)")

    plt.tight_layout()

    _tbl = pl.DataFrame([_stats(trades_model, "model_top10pct"), _stats(trades_random, "random")])
    mo.vstack([fig_diag, _tbl])
    return


@app.cell
def _(TRADE_LOG_CSV, mo, trade_log):
    TRADE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    trade_log.write_csv(TRADE_LOG_CSV)
    mo.md(f"Trade log saved: `{TRADE_LOG_CSV}` ({trade_log.height} rows)")
    return


@app.cell
def _(df, np, oof, pl, tokens):
    _mask_oof = ~np.isnan(oof)
    _scores_valid = oof[_mask_oof]
    _ids_valid = df["token_id"].to_numpy()[_mask_oof]
    _thresh = float(np.quantile(_scores_valid, 0.9))
    _n_buy = int((_scores_valid >= _thresh).sum())

    _oof_out = pl.DataFrame({
        "token_id": pl.Series(_ids_valid, dtype=pl.Int32),
        "oof_score": pl.Series(_scores_valid, dtype=pl.Float64),
    }).with_columns(
        pl.col("oof_score").rank(descending=True).cast(pl.Int32).alias("rank"),
        (pl.col("oof_score") >= _thresh).cast(pl.Int8).alias("buy"),
    )
    _meta = tokens.select("token_id", "name", "ticker")
    scored_df = (
        df.select("token_id", "deploy_time_unix", "hit_2x")
        .join(_oof_out, on="token_id", how="left")
        .join(_meta, on="token_id", how="left")
    )
    top1000 = (
        scored_df.filter(pl.col("buy") == 1)
        .sort("oof_score", descending=True).head(1000)
        .select("rank", "token_id", "name", "ticker", "deploy_time_unix", "oof_score", "hit_2x", "buy")
    )
    thresh_val = _thresh
    n_buy_total = _n_buy
    n_scored = int(_mask_oof.sum())
    return n_buy_total, n_scored, thresh_val, top1000


@app.cell
def _(OUT_CSV, mo, n_buy_total, n_scored, thresh_val, top1000):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    top1000.write_csv(OUT_CSV)
    _hit_rate = top1000["hit_2x"].mean()
    _hr_str = f"{_hit_rate:.3f}" if _hit_rate is not None else "n/a"
    mo.hstack([
        mo.stat(label="scored tokens", value=f"{n_scored:,}"),
        mo.stat(label="buy decisions (p90)", value=f"{n_buy_total:,}"),
        mo.stat(label="threshold", value=f"{thresh_val:.4f}"),
        mo.stat(label="hit_2x in top-1000", value=_hr_str),
    ])
    return


@app.cell
def _(mo, top1000):
    mo.vstack([
        mo.md("**Top 20**"),
        top1000.head(20),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
