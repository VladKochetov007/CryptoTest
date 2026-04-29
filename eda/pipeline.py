"""Pump.fun pipeline: data -> features -> meta__lgbm -> backtest (trailing_30)."""
import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
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
    ART = ROOT / "eda" / "artifacts" / "meta__lgbm"
    SLOTS = ROOT / "slot_features_60m.parquet"
    return ART, Counter, Path, ROOT, SLOTS, dataclass, json, lgb, mo, np, pl, plt, roc_auc_score


@app.cell
def _(ROOT, pl):
    _feat = pl.read_parquet(ROOT / "eda" / "features.parquet").drop_nulls("hit_2x").sort("deploy_time_unix")
    _meta = pl.read_parquet(ROOT / "eda" / "meta_features.parquet")
    df = _feat.join(_meta, on="token_id", how="inner")
    return (df,)


@app.cell
def _():
    BASE_FEATURES = [
        "deployer_deposit_amount", "deployer_wallet_balance_before",
        "deployer_wallet_balance_after_sol", "deployer_wallet_source_amount_sol",
        "is_cex", "deployer_wallet_source_cex_name",
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
    CAT_COL = "deployer_wallet_source_cex_name"
    return BASE_FEATURES, CAT_COL, META_FEATURES


@app.cell
def _(ART, BASE_FEATURES, CAT_COL, META_FEATURES, df, json, lgb, np):
    _all = BASE_FEATURES + META_FEATURES
    model_cols = [c for c in _all if c in df.columns]

    booster = lgb.Booster(model_file=str(ART / "model_last.txt"))
    oof = np.load(ART / "oof_pred.npy")
    _summary = json.loads((ART / "summary.json").read_text())
    fold_aucs = _summary["fold_aucs"]
    return booster, fold_aucs, model_cols, oof


@app.cell
def _(df, fold_aucs, mo, np, oof, pl, roc_auc_score):
    _mask = ~np.isnan(oof)
    oof_auc = roc_auc_score(df["hit_2x"].to_numpy()[_mask].astype(int), oof[_mask])

    _fold_tbl = pl.DataFrame({"fold": list(range(len(fold_aucs))), "auc": fold_aucs})
    mo.hstack([
        mo.stat(label="OOF AUC", value=f"{oof_auc:.4f}"),
        mo.stat(label="n_tokens", value=f"{_mask.sum():,}"),
        mo.stat(label="base_rate", value=f"{df['hit_2x'].mean():.3f}"),
        mo.vstack([mo.md("**Fold AUCs**"), _fold_tbl]),
    ])
    return (oof_auc,)


@app.cell
def _(booster, model_cols, np, plt):
    _gain = booster.feature_importance(importance_type="gain")
    _idx = np.argsort(_gain)[::-1][:30]
    _names = [model_cols[i] for i in _idx]
    _vals = _gain[_idx] / _gain.sum()

    fig_imp, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(30), _vals[::-1])
    ax.set_yticks(range(30))
    ax.set_yticklabels(_names[::-1], fontsize=8.5)
    ax.set_xlabel("gain share")
    ax.set_title("Top 30 features — LightGBM gain")
    plt.tight_layout()
    fig_imp


@app.cell
def _(df, np, oof):
    _mask = ~np.isnan(oof)
    _thresh = np.quantile(oof[_mask], 0.9)
    buy_ids = df["token_id"].to_numpy()[_mask][oof[_mask] >= _thresh].tolist()
    return (buy_ids,)


@app.cell
def _(SLOTS, buy_ids, pl):
    _raw = (
        pl.scan_parquet(SLOTS)
        .filter(pl.col("token_id").is_in(buy_ids) & (pl.col("price_sol_per_token") > 0))
        .select("token_id", "seconds_since_deploy", "price_sol_per_token")
        .sort(["token_id", "seconds_since_deploy"])
        .collect()
    )
    panels = {t: g.sort("seconds_since_deploy") for t, g in _raw.group_by("token_id")}
    return (panels,)


@app.cell
def _(dataclass, panels, np, pl):
    @dataclass
    class Trade:
        token_id: int
        roi: float
        exit_sec: int
        reason: str

    def _trailing_30(tid, secs, prices):
        entry = prices[0]
        cum_max = entry
        armed = False
        for i in range(1, len(prices)):
            cum_max = max(cum_max, prices[i])
            if not armed and prices[i] >= 1.5 * entry:
                armed = True
            if armed and prices[i] <= 0.7 * cum_max:
                return Trade(tid, prices[i] / entry - 1, int(secs[i]), "trail")
            if prices[i] <= 0.4 * entry:
                return Trade(tid, prices[i] / entry - 1, int(secs[i]), "sl_hard")
        return Trade(tid, prices[-1] / entry - 1, int(secs[-1]), "timeout")

    trades = []
    for _tid, _panel in panels.items():
        _sub = _panel.filter(pl.col("seconds_since_deploy") <= 1800)
        if _sub.height == 0:
            continue
        trades.append(_trailing_30(
            _tid,
            _sub["seconds_since_deploy"].to_numpy(),
            _sub["price_sol_per_token"].to_numpy(),
        ))

    rois = np.array([t.roi for t in trades])
    return Trade, rois, trades


@app.cell
def _(Counter, mo, np, plt, rois, trades):
    _sorted = sorted(trades, key=lambda t: t.exit_sec)
    _cum = np.cumsum([t.roi for t in _sorted])
    _reasons = Counter(t.reason for t in trades)

    fig_eq, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(_cum, lw=1.5)
    ax1.axhline(0, color="grey", lw=0.5, ls="--")
    ax1.set_xlabel("trade #")
    ax1.set_ylabel("cumulative ROI (SOL)")
    ax1.set_title(f"trailing_30  n={len(trades)}  mean={rois.mean():.3f}")

    ax2.bar(list(_reasons.keys()), list(_reasons.values()), color="#4878cf")
    ax2.set_title("exit reasons")
    ax2.set_xlabel("reason")
    ax2.set_ylabel("count")

    plt.tight_layout()
    mo.vstack([fig_eq, mo.md(
        f"**win rate** {(rois > 0).mean():.3f}  |  "
        f"**median ROI** {float(np.median(rois)):.4f}  |  "
        f"**p10/p90** {float(np.quantile(rois, .1)):.4f} / {float(np.quantile(rois, .9)):.4f}  |  "
        f"**worst** {float(rois.min()):.4f}"
    )])


if __name__ == "__main__":
    app.run()
