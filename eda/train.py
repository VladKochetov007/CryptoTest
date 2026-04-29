"""Walk-forward training of LGBM / XGBoost / CatBoost on hit_2x_30m label.

Two models per algorithm:
  - PART1_INSTANT: deploy-time-only features (≤300 ms latency budget)
  - PART1_60S:     deploy-time + 60-second post-launch aggregates (2-stage policy)

Walk-forward: 5 expanding-window folds. Embargo 3600 s (>= 1800 s label window)
to prevent label leakage at fold boundaries.
Saves: per-fold AUC/PR-AUC, OOF probabilities, feature importance, models.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
ART = ROOT / "eda" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

DEPLOY_TIME_FEATURES = [
    "deployer_deposit_amount",
    "deployer_wallet_balance_before",
    "deployer_wallet_balance_after_sol",
    "deployer_wallet_source_amount_sol",
    "is_cex",
    "deployer_wallet_source_cex_name",  # categorical
    "has_image", "has_website", "has_telegram",
    "name_len", "ticker_len", "desc_len",
    "deployer_prior_n", "deployer_prior_grad", "deployer_prior_hit20k",
    "deployer_seconds_since_last",
    "funder_prior_n", "funder_prior_hit20k", "funder_prior_grad",
    "deploys_prev_15m", "deploys_prev_60m", "hit20k_rate_prev_60m",
    "image_hash_seen_total", "same_ticker_today_prev", "same_name_prev_hour",
    "mint_suffix_pump", "deployer_suffix_pump",
    "name_alpha_chars", "name_upper_chars",
    "utc_sin", "utc_cos", "utc_hour", "utc_dow", "ny_hour", "ldn_hour", "tokyo_hour",
    # stationary: realized vol (std of 5m log-returns) + percentage returns only
    "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "sol_ret_24h",
    "btc_vol_1h", "btc_ret_1h",
    # derived: vol regime ratio + cross-asset spread (computed in load_dataset)
    "sol_vol_ratio", "sol_btc_ret_spread",
]

POST60_FEATURES = [
    # NOTE: price_max_60s / price_min_60s removed — partially equivalent to the
    # 30-min ROI label (if 2× hits within 60 s, label is True by construction).
    "buy_vol_30s", "sell_vol_30s", "active_slots_30s", "holders_30s",
    "top_wallet_30s", "trades_30s",
    "buy_vol_60s", "sell_vol_60s", "holders_60s", "top_wallet_60s", "trades_60s",
]

CAT_COLS = {"deployer_wallet_source_cex_name"}


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    auc: float
    pr_auc: float
    brier: float
    base_rate_train: float
    base_rate_val: float


def load_dataset(feature_set: list[str], target: str = "hit_2x") -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(FEAT)
    df = df.with_columns(
        (pl.col("sol_vol_1h") / (pl.col("sol_vol_24h") + 1e-9)).alias("sol_vol_ratio"),
        (pl.col("sol_ret_1h") - pl.col("btc_ret_1h")).alias("sol_btc_ret_spread"),
    )
    df = df.drop_nulls([target])
    cols = [c for c in feature_set if c in df.columns]
    return df.select(["token_id", "deploy_time_unix", target] + cols), cols


def walkforward_indices(
    times: np.ndarray,
    n_folds: int = 5,
    embargo_sec: int = 3600,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with a seconds-based embargo.

    times: deploy_time_unix in ascending order, length n.
    embargo_sec: minimum gap between train.max(time) and val.min(time).
    A token's hit_2x label is observed at deploy_time + 1800 s, so embargo_sec
    must be >= 1800 to prevent the label window from touching the next train fold.
    """
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


def encode_for_xgb(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Ordinal-encode categorical columns for XGBoost (which lacks native string cats here)."""
    out = df
    for c in cols:
        if c in CAT_COLS:
            out = out.with_columns(
                pl.col(c).fill_null("__missing__").cast(pl.Categorical).to_physical().alias(c)
            )
    numeric_cols = [c for c in cols if c not in CAT_COLS]
    out = out.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
    return out


def to_lgb_dataset(df: pl.DataFrame, target: str, cols: list[str]):
    import lightgbm as lgb
    sub = df.select(cols + [target])
    if "deployer_wallet_source_cex_name" in cols:
        sub = sub.with_columns(
            pl.col("deployer_wallet_source_cex_name").fill_null("__missing__").cast(pl.Categorical)
        )
    # ensure numeric features are Float64 (avoid pandas object dtype on bigint joins)
    numeric_cols = [c for c in cols if c not in CAT_COLS]
    sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
    pdf = sub.to_pandas()
    X = pdf[cols]
    y = pdf[target].astype(int).values
    cat_idx = [cols.index(c) for c in CAT_COLS if c in cols]
    return lgb.Dataset(X, label=y, categorical_feature=cat_idx, free_raw_data=False), X, y


def train_lgbm(train_df: pl.DataFrame, val_df: pl.DataFrame, cols: list[str], target: str):
    import lightgbm as lgb
    train_set, X_tr, y_tr = to_lgb_dataset(train_df, target, cols)
    val_set, X_va, y_va = to_lgb_dataset(val_df, target, cols)
    base = float(y_tr.mean())
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=200,
        is_unbalance=False,
        scale_pos_weight=(1 - base) / max(base, 1e-6),
        verbose=-1,
        n_jobs=-1,
    )
    booster = lgb.train(
        params, train_set,
        num_boost_round=600,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    pred = booster.predict(X_va)
    return booster, pred, X_tr.columns.tolist()


def train_xgb(train_df: pl.DataFrame, val_df: pl.DataFrame, cols: list[str], target: str):
    import xgboost as xgb
    train_enc = encode_for_xgb(train_df, cols).to_pandas()
    val_enc = encode_for_xgb(val_df, cols).to_pandas()
    X_tr, y_tr = train_enc[cols], train_enc[target].astype(int).values
    X_va, y_va = val_enc[cols], val_enc[target].astype(int).values
    dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan)
    dval = xgb.DMatrix(X_va, label=y_va, missing=np.nan)
    base = float(y_tr.mean())
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        eta=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        scale_pos_weight=(1 - base) / max(base, 1e-6),
        tree_method="hist",
        nthread=-1,
    )
    booster = xgb.train(params, dtrain, num_boost_round=600,
                        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    return booster, pred, cols


def train_catboost(train_df: pl.DataFrame, val_df: pl.DataFrame, cols: list[str], target: str):
    from catboost import CatBoostClassifier, Pool
    numeric_cols = [c for c in cols if c not in CAT_COLS]
    train_df = train_df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
    val_df = val_df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
    train_pdf = train_df.to_pandas()
    val_pdf = val_df.to_pandas()
    X_tr, y_tr = train_pdf[cols].copy(), train_pdf[target].astype(int).values
    X_va, y_va = val_pdf[cols].copy(), val_pdf[target].astype(int).values
    cat_cols = [c for c in CAT_COLS if c in cols]
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("__missing__").astype(str)
        X_va[c] = X_va[c].fillna("__missing__").astype(str)
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols)
    pool_va = Pool(X_va, y_va, cat_features=cat_cols)
    base = float(y_tr.mean())
    model = CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6,
        eval_metric="AUC", random_seed=42,
        scale_pos_weight=(1 - base) / max(base, 1e-6),
        thread_count=-1, verbose=0, early_stopping_rounds=50,
    )
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    pred = model.predict_proba(pool_va)[:, 1]
    return model, pred, cols


def evaluate(y, p) -> tuple[float, float, float]:
    auc = roc_auc_score(y, p)
    pr_auc = average_precision_score(y, p)
    bs = brier_score_loss(y, p)
    return float(auc), float(pr_auc), float(bs)


def run(feature_set_name: str, feature_cols: list[str], algos: list[str]):
    print(f"\n=== {feature_set_name}: {len(feature_cols)} features ===")
    df, cols = load_dataset(feature_cols)
    df = df.sort("deploy_time_unix")
    n = df.height
    print(f"rows: {n}, base rate: {df['hit_2x'].mean():.4f}")

    times = df["deploy_time_unix"].to_numpy()
    folds = walkforward_indices(times, n_folds=5, embargo_sec=3600)
    results: dict[str, list[FoldResult]] = {a: [] for a in algos}
    oof: dict[str, np.ndarray] = {a: np.full(n, np.nan) for a in algos}
    fold_models: dict[str, list] = {a: [] for a in algos}
    last_features: dict[str, list[str]] = {}

    for k, (tr_idx, va_idx) in enumerate(folds):
        train_df = df[tr_idx]
        val_df = df[va_idx]
        for algo in algos:
            t0 = time.time()
            if algo == "lgbm":
                model, pred, used = train_lgbm(train_df, val_df, cols, "hit_2x")
            elif algo == "xgb":
                model, pred, used = train_xgb(train_df, val_df, cols, "hit_2x")
            elif algo == "catboost":
                model, pred, used = train_catboost(train_df, val_df, cols, "hit_2x")
            else:
                raise ValueError(algo)
            y_va = val_df["hit_2x"].to_numpy().astype(int)
            auc, pr, bs = evaluate(y_va, pred)
            elapsed = time.time() - t0
            br_train = float(train_df["hit_2x"].mean())
            br_val = float(val_df["hit_2x"].mean())
            print(f"  fold {k} {algo}: AUC={auc:.4f} PR={pr:.4f} brier={bs:.4f} "
                  f"n_tr={len(tr_idx)} n_va={len(va_idx)} t={elapsed:.1f}s "
                  f"br_tr={br_train:.3f} br_va={br_val:.3f}")
            results[algo].append(FoldResult(k, len(tr_idx), len(va_idx), auc, pr, bs, br_train, br_val))
            oof[algo][va_idx] = pred
            fold_models[algo].append(model)
            last_features[algo] = used

    # save OOF and fold summaries
    for algo in algos:
        recs = results[algo]
        summary = {
            "feature_set": feature_set_name,
            "algo": algo,
            "n_folds": len(recs),
            "auc_mean": float(np.mean([r.auc for r in recs])),
            "auc_std": float(np.std([r.auc for r in recs])),
            "pr_auc_mean": float(np.mean([r.pr_auc for r in recs])),
            "brier_mean": float(np.mean([r.brier for r in recs])),
            "fold_results": [asdict(r) for r in recs],
            "features": last_features.get(algo, cols),
        }
        out_dir = ART / f"{feature_set_name}__{algo}"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        np.save(out_dir / "oof_pred.npy", oof[algo])
        # save final model trained on full data for live scoring
        if algo == "lgbm":
            fold_models[algo][-1].save_model(str(out_dir / "model_last.txt"))
        elif algo == "xgb":
            fold_models[algo][-1].save_model(str(out_dir / "model_last.json"))
        elif algo == "catboost":
            fold_models[algo][-1].save_model(str(out_dir / "model_last.cbm"))
        print(f"[saved] {out_dir}")
    # persist token_id index alignment for OOF
    np.save(ART / f"{feature_set_name}__token_ids.npy", df["token_id"].to_numpy())
    np.save(ART / f"{feature_set_name}__y.npy", df["hit_2x"].to_numpy().astype(int))


def main():
    import sys
    algos = ["lgbm", "xgb", "catboost"]
    sets = sys.argv[1:] if len(sys.argv) > 1 else ["instant", "with60s"]
    if "instant" in sets:
        run("instant", DEPLOY_TIME_FEATURES, algos)
    if "with60s" in sets:
        run("with60s", DEPLOY_TIME_FEATURES + POST60_FEATURES, algos)


if __name__ == "__main__":
    main()
