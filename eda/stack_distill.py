"""Phase 6: Stack base GBMs + meta features, distill to single LGBM for production.

Steps:
1. Load OOF predictions from base models (instant/lgbm, instant/xgb, instant/catboost)
2. Load meta model OOF from meta__lgbm
3. Blend OOF via isotonic-weighted ensemble (geometric mean of calibrated probs)
4. Train distilled LGBM on raw features with blended OOF as soft regression target
5. Evaluate: distilled AUC vs stack AUC vs baseline
6. Save distilled model + calibration for production scoring

Requires:
- eda/artifacts/instant__lgbm/oof_pred.npy
- eda/artifacts/instant__xgb/oof_pred.npy
- eda/artifacts/instant__catboost/oof_pred.npy
- eda/artifacts/meta__lgbm/oof_pred.npy (from meta_train.py)

Outputs:
- eda/artifacts/distilled/model.txt       — production LGBM
- eda/artifacts/distilled/calibrator.pkl  — isotonic calibration
- eda/artifacts/distilled/summary.json
- eda/plots/stack_distill_comparison.png  — AUC comparison chart
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
META = ROOT / "eda" / "meta_features.parquet"
ART = ROOT / "eda" / "artifacts"
OUT_DIR = ART / "distilled"
PLOT_DIR = ROOT / "eda" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

BASE_FEATURES = [
    "deployer_deposit_amount",
    "deployer_wallet_balance_before",
    "deployer_wallet_balance_after_sol",
    "deployer_wallet_source_amount_sol",
    "is_cex",
    "deployer_wallet_source_cex_name",
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

META_FEATURES = [
    "deployer_prior_n_1h", "deployer_prior_n_6h", "deployer_prior_n_24h", "deployer_prior_n_7d",
    "deployer_hr_1h", "deployer_hr_24h", "deployer_hr_7d",
    "funder_prior_n_24h", "funder_hr_24h",
    "handle_len", "handle_digit_ratio", "handle_has_underscore",
    "handle_ends_in_digits", "handle_is_celeb", "handle_contains_ticker",
    "handle_prior_n_24h", "handle_hr_24h",
    "desc_has_url", "desc_has_deployed_template", "desc_exclamation_count",
    "desc_template_score", "desc_word_count", "desc_has_address",
    "name_has_meme_kw", "ticker_has_meme_kw",
    "name_word_count", "name_digit_count", "ticker_digit_count",
    "ticker_special_count", "ticker_len_4_5", "meme_kw_hr_24h",
]

CAT_COL = "deployer_wallet_source_cex_name"


def load_oof(name: str) -> np.ndarray:
    path = ART / name / "oof_pred.npy"
    if not path.exists():
        print(f"  [warn] missing OOF: {path}")
        return None
    return np.load(path)


def calibrate_oof(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(p) & ~np.isnan(y.astype(float))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(p[mask], y[mask].astype(int))
    out = np.full_like(p, np.nan)
    out[mask] = cal.transform(p[mask])
    return out


def geometric_mean_blend(arrays: list[np.ndarray], eps: float = 1e-7) -> np.ndarray:
    """Geometric mean in probability space (equivalent to averaging log-odds)."""
    n = arrays[0].shape[0]
    log_odds_sum = np.zeros(n)
    count = np.zeros(n)
    for a in arrays:
        mask = ~np.isnan(a)
        lo = np.log(np.clip(a, eps, 1 - eps) / np.clip(1 - a, eps, 1 - eps))
        log_odds_sum[mask] += lo[mask]
        count[mask] += 1
    avg_lo = np.where(count > 0, log_odds_sum / count, np.nan)
    return np.where(np.isnan(avg_lo), np.nan, 1 / (1 + np.exp(-avg_lo)))


def walkforward_splits(n: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = np.linspace(0, n, n_folds + 2, dtype=int)
    return [(np.arange(0, edges[k + 1]), np.arange(edges[k + 1], edges[k + 2]))
            for k in range(n_folds)]


def prep_X(df: pl.DataFrame, cols: list[str]) -> "pd.DataFrame":
    sub = df.select(cols)
    if CAT_COL in cols:
        sub = sub.with_columns(
            pl.col(CAT_COL).fill_null("__missing__").cast(pl.Categorical)
        )
    numeric = [c for c in cols if c != CAT_COL]
    sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    return sub.to_pandas()


def train_distilled(df: pl.DataFrame, soft_labels: np.ndarray, cols: list[str]) -> lgb.Booster:
    """Train LGBM to regress onto soft blend targets (distillation)."""
    mask = ~np.isnan(soft_labels)
    df_clean = df[np.where(mask)[0].tolist()]
    y_soft = soft_labels[mask]

    X = prep_X(df_clean, cols)
    cat_idx = [cols.index(CAT_COL)] if CAT_COL in cols else []
    ds = lgb.Dataset(X, label=y_soft, categorical_feature=cat_idx, free_raw_data=False)

    params = dict(
        objective="regression",  # soft targets, not binary
        metric="rmse",
        learning_rate=0.05,
        num_leaves=31,          # smaller — prevent overfit to soft labels
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1, n_jobs=-1,
    )
    booster = lgb.train(params, ds, num_boost_round=300)
    return booster


def plot_auc_comparison(results: dict[str, float]):
    labels = list(results.keys())
    aucs = list(results.values())
    baseline = 0.7653
    colors = ["#e74c3c" if a > baseline + 0.005 else "#3498db" if a > baseline else "#95a5a6"
              for a in aucs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), aucs, color=colors)
    ax.axhline(baseline, color="orange", linestyle="--", linewidth=2, label=f"baseline {baseline:.4f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("OOF AUC")
    ax.set_title("Model comparison: base → meta → stack → distilled")
    ax.legend()
    ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.01)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = PLOT_DIR / "stack_distill_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def main():
    print("[stack_distill] loading features")
    base = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    y = base["hit_2x"].to_numpy().astype(int)

    meta_avail = (ROOT / "eda" / "meta_features.parquet").exists()
    if meta_avail:
        meta = pl.read_parquet(META)
        df = base.join(meta, on="token_id", how="left")
        all_cols = BASE_FEATURES + META_FEATURES
    else:
        print("  [warn] meta_features.parquet not found — using base features only")
        df = base
        all_cols = BASE_FEATURES

    cols = [c for c in all_cols if c in df.columns]
    n = df.height

    print("[stack_distill] loading OOF predictions")
    oof_names = ["instant__lgbm", "instant__xgb", "instant__catboost"]
    oofs = {}
    for name in oof_names:
        raw = load_oof(name)
        if raw is not None and len(raw) == n:
            oofs[name] = calibrate_oof(y, raw)
        else:
            print(f"  [skip] {name}: not found or wrong size")

    meta_oof_raw = load_oof("meta__lgbm")
    if meta_oof_raw is not None and len(meta_oof_raw) == n:
        oofs["meta__lgbm"] = calibrate_oof(y, meta_oof_raw)

    if not oofs:
        print("[error] no OOF predictions available. run train.py and meta_train.py first.")
        return

    print(f"  available OOFs: {list(oofs.keys())}")

    print("[stack_distill] computing OOF AUCs")
    auc_results = {}
    for name, p in oofs.items():
        mask = ~np.isnan(p)
        if mask.sum() < 100:
            continue
        auc = float(roc_auc_score(y[mask].astype(int), p[mask]))
        auc_results[name] = auc
        print(f"  {name}: AUC={auc:.4f}")

    print("[stack_distill] blending")
    blend = geometric_mean_blend(list(oofs.values()))
    mask = ~np.isnan(blend)
    blend_auc = float(roc_auc_score(y[mask].astype(int), blend[mask]))
    print(f"  blended stack AUC: {blend_auc:.4f}")
    auc_results["stack_blend"] = blend_auc

    print("[stack_distill] training distilled LGBM on soft labels")
    distilled = train_distilled(df, blend, cols)
    raw_dist_pred = distilled.predict(prep_X(df, cols))

    # calibrate distilled predictions
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_dist_pred, y)
    dist_pred = cal.transform(raw_dist_pred)

    dist_auc = float(roc_auc_score(y, dist_pred))
    dist_pr = float(average_precision_score(y, dist_pred))
    dist_brier = float(brier_score_loss(y, dist_pred))
    print(f"  distilled AUC: {dist_auc:.4f} (train AUC — not OOF, optimistic)")
    auc_results["distilled_train"] = dist_auc

    # walk-forward OOF for distilled
    print("[stack_distill] walk-forward OOF for distilled model")
    splits = walkforward_splits(n)
    dist_oof = np.full(n, np.nan)
    for k, (tr_idx, va_idx) in enumerate(splits):
        tr_blend = blend[tr_idx]
        va_mask = ~np.isnan(tr_blend)
        if va_mask.sum() < 100:
            continue
        X_tr = prep_X(df[tr_idx], cols)
        X_va = prep_X(df[va_idx], cols)
        y_soft = tr_blend
        cat_idx = [cols.index(CAT_COL)] if CAT_COL in cols else []
        ds = lgb.Dataset(
            prep_X(df[tr_idx][np.where(~np.isnan(tr_blend))[0].tolist()], cols),
            label=tr_blend[~np.isnan(tr_blend)],
            categorical_feature=cat_idx, free_raw_data=False,
        )
        params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                      num_leaves=31, feature_fraction=0.85, bagging_fraction=0.85,
                      bagging_freq=5, min_data_in_leaf=200, verbose=-1, n_jobs=-1)
        b = lgb.train(params, ds, num_boost_round=300)
        pred = b.predict(X_va)
        dist_oof[va_idx] = pred

    dist_oof_mask = ~np.isnan(dist_oof)
    if dist_oof_mask.sum() > 100:
        dist_oof_auc = float(roc_auc_score(y[dist_oof_mask].astype(int), dist_oof[dist_oof_mask]))
        print(f"  distilled OOF AUC: {dist_oof_auc:.4f}")
        auc_results["distilled_oof"] = dist_oof_auc

    # save
    distilled.save_model(str(OUT_DIR / "model.txt"))
    with open(OUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
    np.save(OUT_DIR / "oof_pred.npy", dist_oof)

    summary = {
        "auc_results": auc_results,
        "baseline_lgbm_oof": 0.7653,
        "cols_used": len(cols),
        "blend_sources": list(oofs.keys()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot_auc_comparison(auc_results)
    print(f"[stack_distill] done. artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
