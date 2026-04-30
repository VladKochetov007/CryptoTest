"""Train LGBM on base + meta features. Compare AUC vs current instant baseline.

Requires eda/meta_features.parquet (from build_meta_features.py).

Outputs:
- eda/artifacts/meta__lgbm/summary.json  — fold AUCs, OOF
- eda/artifacts/meta__lgbm/shap_meta.json — top-30 SHAP for new features
- eda/plots/meta_shap.png               — beeswarm for human review
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
META = ROOT / "eda" / "meta_features.parquet"
ART = ROOT / "eda" / "artifacts" / "meta__lgbm"
PLOT_DIR = ROOT / "eda" / "plots"
ART.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "deployer_deposit_amount",
    "deployer_wallet_balance_before",
    "deployer_wallet_balance_after_sol",
    "deployer_wallet_source_amount_sol",
    "is_cex",
    "deployer_wallet_source_cex_name",
    "has_image", "has_website", "has_telegram",
    "name_len", "ticker_len", "desc_len",
    "deployer_prior_n",
    "deployer_seconds_since_last",
    "funder_prior_n",
    "deploys_prev_15m", "deploys_prev_60m",
    "same_ticker_today_prev", "same_name_prev_hour",
    "mint_suffix_pump", "deployer_suffix_pump",
    "name_alpha_chars", "name_upper_chars",
    "utc_sin", "utc_cos", "utc_hour", "utc_dow", "ny_hour", "ldn_hour", "tokyo_hour",
    # stationary: realized vol (std of 5m log-returns) + percentage returns only
    "sol_vol_1h", "sol_vol_24h", "sol_ret_1h", "sol_ret_24h",
    "btc_vol_1h", "btc_ret_1h",
]

META_FEATURES = [
    # multi-scale deployer
    "deployer_prior_n_1h", "deployer_prior_n_6h", "deployer_prior_n_24h", "deployer_prior_n_7d",
    "deployer_hr_1h", "deployer_hr_24h", "deployer_hr_7d",
    # multi-scale funder
    "funder_prior_n_24h", "funder_hr_24h",
    "funder_prior_n_7d", "funder_hr_7d",
    # funder graph
    "funder_seconds_since_last", "funder_unique_deployers_prior",
    "funder_concentration_hhi", "funder_avg_deposit_sol_prior",
    "funder_is_dust_funder",
    # handle sybil
    "handle_unique_deployers_7d",
    # twitter handle static
    "handle_len", "handle_digit_ratio", "handle_has_underscore",
    "handle_ends_in_digits", "handle_is_celeb", "handle_contains_ticker",
    # twitter handle rolling
    "handle_prior_n_24h", "handle_hr_24h",
    # description text
    "desc_has_url", "desc_has_deployed_template", "desc_exclamation_count",
    "desc_template_score", "desc_word_count", "desc_has_address",
    # name/ticker text
    "name_has_meme_kw", "ticker_has_meme_kw",
    "name_word_count", "name_digit_count", "ticker_digit_count",
    "ticker_special_count", "ticker_len_4_5",
    # meme keyword rolling
    "meme_kw_hr_24h",
    # image (leak-free strictly-past count; 0 = no image or first use of this hash)
    "image_hash_prior_count",
    # stationary macro derived (computed in build_meta_features)
    "btc_ret_24h", "sol_vol_ratio", "sol_btc_ret_spread",
]

CAT_COL = "deployer_wallet_source_cex_name"

BASELINE_SUMMARY = ROOT / "eda" / "artifacts" / "instant__lgbm" / "summary.json"


def read_baseline_auc() -> float | None:
    if not BASELINE_SUMMARY.exists():
        return None
    try:
        data = json.loads(BASELINE_SUMMARY.read_text())
        return float(data.get("auc_mean"))
    except Exception:
        return None


def walkforward_splits(
    times: np.ndarray,
    n_folds: int = 5,
    embargo_sec: int = 3600,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware walk-forward identical to train.walkforward_indices."""
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


def prep(df: pl.DataFrame, cols: list[str], target: str):
    sub = df.select(cols + [target])
    sub = sub.with_columns(
        pl.col(CAT_COL).fill_null("__missing__").cast(pl.Categorical)
        if CAT_COL in cols else pl.lit(None)
    )
    numeric = [c for c in cols if c != CAT_COL]
    sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    pdf = sub.to_pandas()
    X = pdf[cols]
    y = pdf[target].astype(int).values
    cat_idx = [cols.index(CAT_COL)] if CAT_COL in cols else []
    return X, y, cat_idx


def main():
    baseline_auc = read_baseline_auc()
    print("[meta_train] loading features")
    base = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    meta = pl.read_parquet(META)
    df = base.join(meta, on="token_id", how="left")
    print(f"  rows: {df.height}, base_rate: {df['hit_2x'].mean():.4f}")

    all_cols = BASE_FEATURES + META_FEATURES
    cols = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  [warn] missing columns: {missing}")

    n = df.height
    times = df["deploy_time_unix"].to_numpy()
    splits = walkforward_splits(times, n_folds=5, embargo_sec=3600)
    oof = np.full(n, np.nan)
    fold_aucs = []

    for k, (tr_idx, va_idx) in enumerate(splits):
        tr, va = df[tr_idx], df[va_idx]
        X_tr, y_tr, cat_idx = prep(tr, cols, "hit_2x")
        X_va, y_va, _ = prep(va, cols, "hit_2x")
        base_rate = float(y_tr.mean())

        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, free_raw_data=False)

        t0 = time.time()
        params = dict(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            min_data_in_leaf=200,
            scale_pos_weight=(1 - base_rate) / max(base_rate, 1e-6),
            verbose=-1, n_jobs=-1,
        )
        booster = lgb.train(
            params, ds_tr, num_boost_round=600,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = booster.predict(X_va)
        oof[va_idx] = pred
        auc = float(roc_auc_score(y_va, pred))
        pr_auc = float(average_precision_score(y_va, pred))
        fold_aucs.append(auc)
        print(f"  fold {k}: AUC={auc:.4f} PR={pr_auc:.4f} n_tr={len(tr_idx)} n_va={len(va_idx)} "
              f"t={time.time()-t0:.1f}s br_tr={base_rate:.3f} br_va={float(y_va.mean()):.3f}")

    mask = ~np.isnan(oof)
    oof_auc = float(roc_auc_score(df["hit_2x"].to_numpy()[mask].astype(int), oof[mask]))
    base_str = f"{baseline_auc:.4f}" if baseline_auc is not None else "n/a"
    print(f"\n  OOF AUC: {oof_auc:.4f}  (baseline instant/lgbm: {base_str})")

    # SHAP on last fold model
    print("[meta_train] computing SHAP on last-fold model (20k sample)")
    try:
        import shap
        sample_idx = np.random.choice(np.where(mask)[0], size=min(20000, mask.sum()), replace=False)
        X_sample, _, _ = prep(df[sample_idx.tolist()], cols, "hit_2x")
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X_sample)
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_rank = sorted(zip(cols, mean_abs_shap), key=lambda x: -x[1])

        print("\nTop-20 features by |SHAP|:")
        for name, val in shap_rank[:20]:
            print(f"  {val:.4f}  {name}")

        shap_json = [{"feature": n, "mean_abs_shap": float(v)}
                     for n, v in shap_rank]
        (ART / "shap_meta.json").write_text(json.dumps(shap_json, indent=2))

        # plot: top-30 bar chart
        top_n = shap_rank[:30]
        names = [n for n, _ in top_n]
        vals = [v for _, v in top_n]
        colors = ["#3498db"] * len(names)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title("Top-30 features by mean |SHAP|")
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "meta_shap_importance.png", dpi=150)
        plt.close(fig)
        print(f"[plot] saved {PLOT_DIR}/meta_shap_importance.png")
    except ImportError:
        print("  [skip] shap not installed")

    np.save(ART / "oof_pred.npy", oof)
    booster.save_model(str(ART / "model_last.txt"))
    np.save(ART.parent / "meta__token_ids.npy", df["token_id"].to_numpy())
    np.save(ART.parent / "meta__y.npy", df["hit_2x"].to_numpy().astype(int))
    summary = {
        "feature_set": "meta",
        "algo": "lgbm",
        "n_features_base": len(BASE_FEATURES),
        "n_features_meta": len(META_FEATURES),
        "n_features_used": len(cols),
        "oof_auc": oof_auc,
        "baseline_oof_auc": baseline_auc,
        "delta_auc": (oof_auc - baseline_auc) if baseline_auc is not None else None,
        "fold_aucs": fold_aucs,
        "features": cols,
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
