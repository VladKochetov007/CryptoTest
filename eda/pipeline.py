"""Pump.fun pipeline: tokens.parquet + features.parquet -> meta features -> model -> backtest -> ranked CSV."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import time
    import json
    from collections import Counter
    from dataclasses import dataclass
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

    # make eda/ importable so we can reuse build_meta_features and meta_train helpers
    sys.path.insert(0, str(ROOT / "eda"))
    from build_meta_features import (
        compute_rolling_group_features,
        funder_graph_features,
        twitter_handle_reuse_features,
        twitter_handle_features,
        description_features,
        name_text_features,
        image_hash_rolling_count,
        macro_derived_features,
        meme_kw_rolling_win_rate,
    )
    from meta_train import prep, walkforward_splits

    return (
        Counter,
        OUT_CSV,
        ROOT,
        SLOTS,
        compute_rolling_group_features,
        dataclass,
        description_features,
        funder_graph_features,
        image_hash_rolling_count,
        lgb,
        macro_derived_features,
        meme_kw_rolling_win_rate,
        mo,
        name_text_features,
        np,
        pl,
        plt,
        prep,
        roc_auc_score,
        time,
        twitter_handle_features,
        twitter_handle_reuse_features,
        walkforward_splits,
    )


@app.cell
def _(mo):
    mo.md("""
    # Pump.fun pre-buy scoring pipeline

    End-to-end notebook: raw parquet files → meta feature engineering → LightGBM → backtest → ranked submission CSV.

    **Label**: `hit_2x` — token reached 2× its launch market-cap within the first hour.
    **Base rate**: ~15% across the dataset.
    **Model**: `meta__lgbm` — 84 features (41 base + 43 meta), 5-fold expanding walk-forward with 1h embargo.
    **OOF AUC**: 0.7977 (vs instant baseline 0.7653, +324 bps).
    """)
    return


@app.cell
def _(ROOT, pl):
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
    mo.md(f"**Data ready** — {df.height:,} labeled tokens, meta features in {meta_elapsed:.1f}s")
    return (df,)


@app.cell
def _():
    BASE_FEATURES = [
        "deployer_deposit_amount",
        "deployer_wallet_balance_before",
        "deployer_wallet_balance_after_sol",
        "deployer_wallet_source_amount_sol",
        "is_cex",
        "deployer_wallet_source_cex_name",
        "has_image", "has_website", "has_telegram",
        "name_len", "ticker_len", "desc_len",
        "deployer_prior_n", "deployer_prior_grad", "deployer_prior_hit20k",
        "deployer_seconds_since_last",
        "funder_prior_n", "funder_prior_hit20k", "funder_prior_grad",
        "deploys_prev_15m", "deploys_prev_60m", "hit20k_rate_prev_60m",
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
    return BASE_FEATURES, META_FEATURES


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature reference

    All 84 features used by `meta__lgbm`. SHAP rank = mean |SHAP| rank on last OOF fold (2k sample).

    ### Deployer wallet — on-chain economics (BASE)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `deployer_wallet_balance_after_sol` | 3 | + | SOL balance remaining after funding the token launch. Wealthy deployers tend to be more committed. |
    | `deployer_deposit_amount` | 4 | + | SOL deposited to bonding curve at launch. Higher deposit = more skin in the game. |
    | `deployer_wallet_balance_before` | 13 | + | Pre-deposit wallet balance. Proxy for deployer net worth. |
    | `deployer_wallet_source_amount_sol` | 20 | + | SOL amount sent by the funding wallet to the deployer. Low amount = drip-funder mule pattern. |
    | `is_cex` | — | + | Binary: funder wallet is a known CEX hot wallet. Weak as boolean; `deployer_wallet_source_cex_name` carries the real signal. |
    | `deployer_wallet_source_cex_name` | — | varies | Categorical CEX name. MEXC: +0.50 lift vs base; Gate.io: +1.41 lift. Binance/OKX near neutral. |

    ### Token metadata (BASE)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `has_image` | 11 | + | Token has an uploaded image. 0 = likely bot/spam. SHAP ~0.06. |
    | `has_website` | — | + | Token has a website URL. |
    | `has_telegram` | — | + | Token has a Telegram link. |
    | `name_len` | — | mixed | Character length of token name. Very short or very long names are noisy. |
    | `ticker_len` | — | — | Character length of ticker symbol. |
    | `desc_len` | — | + | Character length of description. 0 = no description (bot signal). Continuous replacement for `has_desc`. |
    | `name_alpha_chars` | — | + | Count of alphabetic characters in name. |
    | `name_upper_chars` | 6 | + | Count of uppercase characters in name. Strong signal — SHAP 0.117. Reflects naming style patterns of successful deployers. |
    | `mint_suffix_pump` | 12 | + | Token mint address ends in "pump". SHAP ~0.059. Pump.fun generated mints always end in "pump"; some deployers use vanity mints. |
    | `deployer_suffix_pump` | — | + | Deployer address ends in "pump". Vanity address signal. |

    ### Deployer history (BASE — instant window at deploy time)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `deployer_prior_n` | 18 | + | Total prior token count for this deployer (all-time). |
    | `deployer_prior_grad` | — | + | Prior graduation count for this deployer. Graduation = market-cap ≥ $69k. |
    | `deployer_prior_hit20k` | 14 | + | Prior count where market-cap exceeded $20k. Leading indicator of quality. |
    | `deployer_seconds_since_last` | 2 | — | Seconds since this deployer's previous token. Short gaps = serial spammer; very long gaps = infrequent = often quality. #2 by SHAP (0.381). |

    ### Funder history (BASE — instant window)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `funder_prior_n` | 16 | + | Prior token count funded by this wallet. |
    | `funder_prior_hit20k` | — | + | Prior count hitting $20k for this funder. |
    | `funder_prior_grad` | — | + | Prior graduation count for this funder. |

    ### Market activity at deploy time (BASE)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `deploys_prev_15m` | — | — | Platform-wide deploy count in last 15 minutes. High = crowded market. |
    | `deploys_prev_60m` | — | — | Platform-wide deploy count in last 60 minutes. |
    | `hit20k_rate_prev_60m` | — | + | Platform-wide hit_20k rate over last 60 minutes. Regime indicator. |
    | `same_ticker_today_prev` | — | — | Prior tokens today with the same ticker. Clone signal. |
    | `same_name_prev_hour` | — | — | Prior tokens in the last hour with the same name. |

    ### Clock features (BASE)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `utc_sin`, `utc_cos` | — | cyclic | Hour-of-day encoded as sin/cos for smooth cyclical representation. |
    | `utc_hour` | — | cyclic | UTC hour integer. |
    | `utc_dow` | — | cyclic | UTC day of week (0=Mon). |
    | `ny_hour` | — | cyclic | Hour in New York timezone (US market hours). |
    | `ldn_hour` | — | cyclic | Hour in London timezone (EU market hours). |
    | `tokyo_hour` | — | cyclic | Hour in Tokyo timezone (Asian market hours). |

    ### Stationary macro — realized volatility and returns (BASE)

    All price-level features (btc_close, sol_close) are excluded — non-stationary I(1) processes that encode bull/bear regime, not a generalizing signal.

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `sol_vol_1h` | — | — | SOL realized vol: std of 5-minute log-returns over last 1h. High vol = choppy market. |
    | `sol_vol_24h` | — | — | SOL realized vol over last 24h. |
    | `sol_ret_1h` | — | + | SOL 1h percentage return. |
    | `sol_ret_24h` | — | + | SOL 24h percentage return. |
    | `btc_vol_1h` | — | — | BTC realized vol over 1h. |
    | `btc_ret_1h` | — | + | BTC 1h percentage return. |

    ---

    ### Multi-scale deployer history (META — rolling windows, O(N log N) per group)

    Computed via `compute_rolling_group_features`: cumsum + searchsorted, strictly past (side="left" on hi), tie-safe.

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `deployer_hr_7d` | **1** | + | Fraction of this deployer's own prior tokens (past 7 days) that hit 2×. Top feature by SHAP (0.693). Deployers who consistently launch successful tokens keep doing so. |
    | `deployer_hr_24h` | 5 | + | Same hit rate over the past 24h only — captures recent hot-streak or cool-off. SHAP 0.131. |
    | `deployer_hr_1h` | 8 | + | Same hit rate over the past 1h — very fresh momentum signal. SHAP 0.064. |
    | `deployer_prior_n_7d` | 9 | + | Deployer token count in last 7 days. Active deployers. SHAP 0.061. |
    | `deployer_prior_n_24h` | — | + | Deployer token count in last 24h. |
    | `deployer_prior_n_6h` | — | + | Deployer token count in last 6h. |
    | `deployer_prior_n_1h` | — | + | Deployer token count in last 1h. |

    ### Multi-scale funder history (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `funder_hr_7d` | 15 | + | Fraction of tokens funded by this wallet (past 7 days) that hit 2×. SHAP 0.054. Smart-money funder signal. |
    | `funder_hr_24h` | — | + | Same hit rate over the past 24h. |
    | `funder_prior_n_7d` | — | + | Funder token count in last 7 days. |
    | `funder_prior_n_24h` | — | + | Funder token count in last 24h. |

    ### Funder graph features (META — past-only snapshot per token)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `funder_seconds_since_last` | — | — | Seconds since funder's previous funding event. Rapid-fire funders are spammers. |
    | `funder_unique_deployers_prior` | — | mixed | Distinct deployer addresses this funder has funded (all past). |
    | `funder_concentration_hhi` | — | — | HHI = Σ(pᵢ²) over deployer-share distribution. HHI=1 means one deployer monopolizes this funder (sybil signal). |
    | `funder_avg_deposit_sol_prior` | — | + | Mean SOL amount deposited by this funder historically. |
    | `funder_is_dust_funder` | — | — | 1 if deposit < 0.5 SOL AND prior_n > 5. Drip-fund mule wallet pattern. |

    ### Twitter handle — sybil and quality signals (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `handle_hr_24h` | 7 | + | Fraction of prior tokens that used this twitter handle (past 24h) and hit 2×. SHAP 0.101. |
    | `handle_prior_n_24h` | — | + | Token count for this handle, last 24h. |
    | `handle_unique_deployers_7d` | — | — | Distinct deployer addresses using this handle in last 7 days. >1 = handle shared across wallets (sybil signal). |
    | `handle_len` | — | + | Character length of twitter handle. 0 = no handle. Continuous replacement for has_twitter. |
    | `handle_digit_ratio` | — | — | Fraction of digits in handle. High ratio = generated/bot handle. |
    | `handle_has_underscore` | — | mixed | Handle contains underscore. |
    | `handle_ends_in_digits` | — | — | Handle ends in digits. Bot-name pattern. |
    | `handle_is_celeb` | — | mixed | Handle matches a known celebrity (Elon, Trump, Vitalik, etc.). Often impersonation. |
    | `handle_contains_ticker` | — | + | Handle starts or ends with the token's ticker symbol. Authentic branding signal. |

    ### Description text features (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `desc_word_count` | — | + | Word count of token description. 0 = no description. |
    | `desc_has_url` | — | mixed | Description contains a URL. |
    | `desc_has_deployed_template` | — | — | Description contains "deployed using" — pump.fun auto-generated template. Quality-negative signal. |
    | `desc_exclamation_count` | — | — | Exclamation marks in description. Hype-language indicator. |
    | `desc_template_score` | — | — | Count of known hype phrases ("100x", "safe", "no rug", etc.) in description. Higher = more copy-paste spam. |
    | `desc_has_address` | — | mixed | Description contains a base58 address string. |

    ### Name/ticker text features (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `name_word_count` | 19 | + | Word count in token name. Multi-word names often more intentional. SHAP 0.039. |
    | `name_digit_count` | — | — | Digit count in name. Random-generated names often contain digits. |
    | `name_has_meme_kw` | — | mixed | Name contains a meme keyword (doge, pepe, moon, etc.). Low IV — very common, not selective. |
    | `ticker_has_meme_kw` | — | mixed | Ticker contains a meme keyword. |
    | `ticker_digit_count` | — | — | Digit count in ticker. |
    | `ticker_special_count` | — | — | Special character count in ticker. |
    | `ticker_len_4_5` | — | + | Ticker length is 4 or 5 characters. Common among legitimate tokens. |

    ### Meme keyword regime (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `meme_kw_hr_24h` | 17 | + | Rolling 24h hit_2x rate for the meme-keyword category (15-min bucket aggregation, shift by 1 bucket). Regime signal: when meme tokens are hot, the signal lifts. SHAP 0.049. |

    ### Image reuse (META)

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `image_hash_prior_count` | 10 | — | Count of prior tokens using the same SHA256 image hash. 0 = no image or first use. Recycled images = serial copycats. SHAP 0.060. |

    ### Stationary macro derived (META)

    Computed in `build_meta_features.macro_derived_features` from already-stationary inputs.

    | Feature | SHAP rank | Direction | Description |
    |---|---|---|---|
    | `btc_ret_24h` | — | + | BTC 24h percentage return (via join_asof on token timestamps — many tokens share 5-min bars). |
    | `sol_vol_ratio` | — | mixed | sol_vol_1h / sol_vol_24h. Short-term vs long-term vol ratio. >1 = vol spike (intraday). |
    | `sol_btc_ret_spread` | — | + | sol_ret_1h − btc_ret_1h. SOL excess return over BTC at 1h horizon. |
    """)
    return


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
    oof_auc = roc_auc_score(df["hit_2x"].to_numpy()[_mask].astype(int), oof[_mask])

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
    _vals = _gain[_idx] / _gain.sum()

    fig_imp, _ax = plt.subplots(figsize=(10, 10))
    _ax.barh(range(30), _vals[::-1])
    _ax.set_yticks(range(30))
    _ax.set_yticklabels(_names[::-1], fontsize=8.5)
    _ax.set_xlabel("gain share")
    _ax.set_title("Top 30 features — LightGBM gain")
    plt.tight_layout()
    fig_imp
    return


@app.cell
def _(mo):
    mo.md("""
    ## Transaction cost model

    Pump.fun uses a **virtual constant-product bonding curve** (`k = v_sol × v_token`).

    - **1% fee** on every buy (deducted from SOL input before the curve) and every sell (deducted from SOL output after the curve).
    - **AMM price impact** for a 0.1 SOL probe at genesis reserves (v_sol=30, v_token=1.073B) ≈ 0.17%; for a 1 SOL full position ≈ 1.7%.
    - **Roundtrip cost model**: `ENTRY_SLIP = 0.013` (1% fee + ~0.3% AMM impact), `EXIT_SLIP = 0.013`.
    - All ROI numbers below are **gross** (no costs applied in the backtest sim); net ROI = gross ROI − 2.6% roundtrip.
    - The AMM two-stage simulation (`eda/two_stage_sim.py`) with `fill_mode="amm"` shows net PnL turns negative on an unfiltered 20k-token universe — confirming that model selection is load-bearing.
    """)
    return


@app.cell
def _():
    ENTRY_SLIP = 0.013
    EXIT_SLIP = 0.013
    return ENTRY_SLIP, EXIT_SLIP


@app.cell
def _(df, np, oof):
    _mask = ~np.isnan(oof)
    _thresh = np.quantile(oof[_mask], 0.9)

    _valid_ids = df["token_id"].to_numpy()[_mask]
    _valid_scores = oof[_mask]

    model_ids = _valid_ids[_valid_scores >= _thresh].tolist()

    rng = np.random.default_rng(42)
    random_ids = rng.choice(_valid_ids, size=len(model_ids), replace=False).tolist()
    return model_ids, random_ids


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
    # group_by yields tuple keys even for single column — unpack with [0]
    panels = {t[0]: g.sort("seconds_since_deploy") for t, g in _raw.group_by("token_id")}
    return (panels,)


@app.cell
def _(ENTRY_SLIP, EXIT_SLIP, dataclass, model_ids, np, panels, pl, random_ids):
    @dataclass
    class Trade:
        token_id: int
        roi: float
        exit_sec: int
        reason: str

    def sim_trailing(tid, secs, prices, entry_slip, exit_slip):
        entry = prices[0] * (1 + entry_slip)
        cum_max = prices[0]
        armed = False
        for i in range(1, len(prices)):
            cum_max = max(cum_max, prices[i])
            if not armed and prices[i] >= 1.5 * prices[0]:
                armed = True
            if armed and prices[i] <= 0.7 * cum_max:
                fill = prices[i] * (1 - exit_slip)
                return Trade(tid, fill / entry - 1, int(secs[i]), "trail")
            if prices[i] <= 0.4 * prices[0]:
                fill = prices[i] * (1 - exit_slip)
                return Trade(tid, fill / entry - 1, int(secs[i]), "sl_hard")
        fill = prices[-1] * (1 - exit_slip)
        return Trade(tid, fill / entry - 1, int(secs[-1]), "timeout")

    def run_backtest(ids, panels, entry_slip, exit_slip):
        trades = []
        for tid in ids:
            panel = panels.get(tid)
            if panel is None:
                continue
            sub = panel.filter(pl.col("seconds_since_deploy") <= 1800)
            if sub.height == 0:
                continue
            trades.append(sim_trailing(
                tid,
                sub["seconds_since_deploy"].to_numpy(),
                sub["price_sol_per_token"].to_numpy(),
                entry_slip, exit_slip,
            ))
        return trades

    trades_model = run_backtest(model_ids, panels, ENTRY_SLIP, EXIT_SLIP)
    trades_random = run_backtest(random_ids, panels, ENTRY_SLIP, EXIT_SLIP)

    rois_m = np.array([t.roi for t in trades_model])
    rois_r = np.array([t.roi for t in trades_random])
    return rois_m, rois_r, trades_model, trades_random


@app.cell
def _(
    Counter,
    ENTRY_SLIP,
    EXIT_SLIP,
    mo,
    np,
    plt,
    rois_m,
    rois_r,
    trades_model,
    trades_random,
):
    _sm = sorted(trades_model, key=lambda t: t.exit_sec)
    _sr = sorted(trades_random, key=lambda t: t.exit_sec)

    _wealth_m = np.cumprod(1 + np.array([t.roi for t in _sm]))
    _wealth_r = np.cumprod(1 + np.array([t.roi for t in _sr]))

    _reasons_m = Counter(t.reason for t in trades_model)
    _reasons_r = Counter(t.reason for t in trades_random)

    fig_eq, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(_wealth_m, lw=1.8, label=f"model top-10% (n={len(trades_model)})", color="#2563eb")
    ax1.plot(_wealth_r, lw=1.2, label=f"random (n={len(trades_random)})", color="#dc2626", alpha=0.7)
    ax1.axhline(1.0, color="grey", lw=0.7, ls="--")
    ax1.set_yscale("log")
    ax1.set_xlabel("trade # (sorted by exit time)")
    ax1.set_ylabel("cumulative wealth (SOL per SOL, log scale)")
    ax1.set_title("Backtest equity — trailing_30 with pump.fun costs")
    ax1.legend()

    _cost_txt = (
        f"Cost model\n"
        f"  entry slip: {ENTRY_SLIP*100:.1f}%\n"
        f"  exit slip:  {EXIT_SLIP*100:.1f}%\n"
        f"  (1% pump fee + ~0.3% AMM impact)"
    )
    ax1.text(0.02, 0.03, _cost_txt, transform=ax1.transAxes,
             fontsize=8, va="bottom", family="monospace",
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    _all_reasons = sorted(set(_reasons_m) | set(_reasons_r))
    _x = np.arange(len(_all_reasons))
    _w = 0.35
    ax2.bar(_x - _w/2, [_reasons_m.get(r, 0) for r in _all_reasons], _w, label="model", color="#2563eb")
    ax2.bar(_x + _w/2, [_reasons_r.get(r, 0) for r in _all_reasons], _w, label="random", color="#dc2626", alpha=0.8)
    ax2.set_xticks(_x)
    ax2.set_xticklabels(_all_reasons)
    ax2.set_title("Exit reason distribution")
    ax2.set_ylabel("count")
    ax2.legend()

    plt.tight_layout()

    def _fmt(arr, fn):
        return f"{fn(arr):.3f}" if len(arr) > 0 else "n/a"

    _net_m = float(np.sum(rois_m)) if len(rois_m) > 0 else 0.0
    _net_r = float(np.sum(rois_r)) if len(rois_r) > 0 else 0.0

    mo.vstack([
        fig_eq,
        mo.md(f"""
    ### Backtest summary (trailing_30, 30-min window, net of {(ENTRY_SLIP+EXIT_SLIP)*100:.1f}% roundtrip cost)

    |  | Model top-10% | Random |
    |--|--|--|
    | n trades | {len(trades_model)} | {len(trades_random)} |
    | win rate | {_fmt(rois_m, lambda a: (a > 0).mean())} | {_fmt(rois_r, lambda a: (a > 0).mean())} |
    | mean ROI | {_fmt(rois_m, np.mean)} | {_fmt(rois_r, np.mean)} |
    | median ROI | {_fmt(rois_m, np.median)} | {_fmt(rois_r, np.median)} |
    | p10 / p90 | {_fmt(rois_m, lambda a: np.quantile(a,.1))} / {_fmt(rois_m, lambda a: np.quantile(a,.9))} | {_fmt(rois_r, lambda a: np.quantile(a,.1))} / {_fmt(rois_r, lambda a: np.quantile(a,.9))} |
    | total PnL (SOL) | {_net_m:.2f} | {_net_r:.2f} |
    | worst trade | {_fmt(rois_m, np.min)} | {_fmt(rois_r, np.min)} |
    """),
    ])
    return


@app.cell
def _(df, np, oof, pl, tokens):
    _mask_oof = ~np.isnan(oof)
    _scores_valid = oof[_mask_oof]
    _ids_valid = df["token_id"].to_numpy()[_mask_oof]

    # threshold = 90th percentile → top 10% of OOF-scored tokens get decision=1
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

    # top 1000 buy tokens for submission
    top1000 = (
        scored_df
        .filter(pl.col("buy") == 1)
        .sort("oof_score", descending=True)
        .head(1000)
        .select("rank", "token_id", "name", "ticker", "deploy_time_unix",
                "oof_score", "hit_2x", "buy")
    )

    thresh_val = _thresh
    n_buy_total = _n_buy
    n_scored = int(_mask_oof.sum())
    return n_buy_total, n_scored, thresh_val, top1000


@app.cell
def _(OUT_CSV, mo, n_buy_total, n_scored, thresh_val, top1000):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    top1000.write_csv(OUT_CSV)

    _hit_rate_buy = top1000["hit_2x"].mean() if top1000["hit_2x"].null_count() < top1000.height else None
    _hr_str = f"{_hit_rate_buy:.3f}" if _hit_rate_buy is not None else "n/a"

    mo.vstack([
        mo.hstack([
            mo.stat(label="scored tokens", value=f"{n_scored:,}"),
            mo.stat(label="buy decisions (p90 threshold)", value=f"{n_buy_total:,}"),
            mo.stat(label="threshold score", value=f"{thresh_val:.4f}"),
            mo.stat(label="hit_2x rate in top-1000", value=_hr_str),
        ]),
        mo.md(f"""
    **Top 1000 by OOF score → `{OUT_CSV.name}`**

    `buy=1` requires `oof_score ≥ {thresh_val:.4f}` (90th percentile of {n_scored:,} OOF-scored tokens).
    The model selects {n_buy_total:,} / {n_scored:,} = **{n_buy_total/n_scored*100:.1f}%** of the full labeled corpus.
    Population base rate is ~15%; hit rate in the top-1000 is **{_hr_str}**.

    `oof_score` = unbiased estimate from the fold where each token was held out (not used in training).
    `hit_2x` = observed ground truth, never used as a feature.
    """),
    ])
    return


@app.cell
def _(mo, top1000):
    mo.vstack([
        mo.md("**Top 20 — highest model confidence**"),
        top1000.head(20),
        mo.md("**Bottom 20 of top-1000 — threshold boundary**"),
        top1000.tail(20),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
