"""Deployer-grouped CV robustness test.

The 8,465-token top deployer dominates the dataset. If the alpha is mostly
"recognise this deployer", a deployer-grouped split should crater AUC. If the
alpha is real (signal in the *features*), AUC should be close to time-fold AUC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
OUT = ROOT / "eda" / "robustness"
OUT.mkdir(exist_ok=True)

CAT_COLS = {"deployer_wallet_source_cex_name"}
DEPLOY_TIME_FEATURES = [
    "deployer_deposit_amount", "deployer_wallet_balance_before",
    "deployer_wallet_balance_after_sol", "deployer_wallet_source_amount_sol",
    "is_cex", "deployer_wallet_source_cex_name",
    "has_image", "has_desc", "has_website", "has_twitter", "has_telegram",
    "name_len", "ticker_len", "desc_len",
    "deployer_prior_n", "deployer_prior_grad", "deployer_prior_hit20k",
    "deployer_seconds_since_last",
    "funder_prior_n", "funder_prior_hit20k", "funder_prior_grad",
    "deploys_prev_15m", "deploys_prev_60m", "hit20k_rate_prev_60m",
    "image_hash_seen_total", "same_ticker_today_prev", "same_name_prev_hour",
    "mint_suffix_pump", "mint_suffix_PUMP", "deployer_suffix_pump",
    "name_alpha_chars", "name_upper_chars",
    "utc_sin", "utc_cos", "utc_hour", "utc_dow", "ny_hour", "ldn_hour", "tokyo_hour",
    "sol_close", "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "sol_ret_24h",
    "btc_close", "btc_vol_1h", "btc_ret_1h",
]


def to_pdf(df: pl.DataFrame, cols: list[str]):
    sub = df.select(cols + ["hit_2x"])
    if "deployer_wallet_source_cex_name" in cols:
        sub = sub.with_columns(
            pl.col("deployer_wallet_source_cex_name").fill_null("__missing__").cast(pl.Categorical)
        )
    numeric_cols = [c for c in cols if c not in CAT_COLS]
    sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
    pdf = sub.to_pandas()
    return pdf[cols], pdf["hit_2x"].astype(int).values


def train_lgbm(X_tr, y_tr, X_va, y_va, cols: list[str]):
    import lightgbm as lgb
    cat_idx = [cols.index(c) for c in CAT_COLS if c in cols]
    train = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    val = lgb.Dataset(X_va, y_va, categorical_feature=cat_idx, free_raw_data=False)
    base = float(y_tr.mean())
    params = dict(
        objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        min_data_in_leaf=200, scale_pos_weight=(1 - base) / max(base, 1e-6),
        verbose=-1, n_jobs=-1,
    )
    booster = lgb.train(params, train, num_boost_round=600, valid_sets=[val],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return booster.predict(X_va)


def main():
    df = pl.read_parquet(FEAT).drop_nulls(["hit_2x"])
    print(f"[robust] rows={df.height}, deployers={df['deployer_address'].n_unique()}")
    cols = [c for c in DEPLOY_TIME_FEATURES if c in df.columns]
    pdf_X, y = to_pdf(df, cols)
    groups = df["deployer_address"].to_numpy()

    fold_results = []
    gkf = GroupKFold(n_splits=5)
    for k, (tr, va) in enumerate(gkf.split(pdf_X, y, groups=groups)):
        pred = train_lgbm(pdf_X.iloc[tr], y[tr], pdf_X.iloc[va], y[va], cols)
        auc = roc_auc_score(y[va], pred)
        pr = average_precision_score(y[va], pred)
        n_dep_tr = len(np.unique(groups[tr]))
        n_dep_va = len(np.unique(groups[va]))
        print(f"  fold {k}: AUC={auc:.4f} PR={pr:.4f} n_tr={len(tr)} n_va={len(va)} "
              f"deployers_tr={n_dep_tr} deployers_va={n_dep_va} "
              f"base_va={y[va].mean():.3f}")
        fold_results.append({"fold": k, "auc": auc, "pr_auc": pr,
                             "n_tr": int(len(tr)), "n_va": int(len(va)),
                             "n_deployers_tr": int(n_dep_tr),
                             "n_deployers_va": int(n_dep_va),
                             "base_rate_va": float(y[va].mean())})
    summary = {
        "auc_mean": float(np.mean([r["auc"] for r in fold_results])),
        "auc_std": float(np.std([r["auc"] for r in fold_results])),
        "pr_auc_mean": float(np.mean([r["pr_auc"] for r in fold_results])),
        "fold_results": fold_results,
    }
    (OUT / "deployer_grouped_cv.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
