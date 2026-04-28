"""Label-shuffle permutation test on the meta feature set.

Shuffles `hit_2x` within each fold's training set, retrains LGBM, scores on
unchanged validation labels. Expected AUC distribution under H0 (no real signal):
mean ≈ 0.50, 95% CI inside [0.48, 0.52]. Anything substantially above 0.52
indicates residual leakage.

Output: eda/artifacts/meta__lgbm/permutation_null.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from meta_train import BASE_FEATURES, META_FEATURES, prep, walkforward_splits

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
META = ROOT / "eda" / "meta_features.parquet"
ART = ROOT / "eda" / "artifacts" / "meta__lgbm"

N_SHUFFLES = 5  # 5 shuffles × 5 folds = 25 AUC samples, enough for a tight CI


def main():
    print("[perm] loading features")
    base = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    meta = pl.read_parquet(META)
    df = base.join(meta, on="token_id", how="left")
    cols = [c for c in BASE_FEATURES + META_FEATURES if c in df.columns]
    times = df["deploy_time_unix"].to_numpy()
    splits = walkforward_splits(times, n_folds=5, embargo_sec=3600)

    rng = np.random.default_rng(42)
    aucs: list[float] = []

    for shuffle_idx in range(N_SHUFFLES):
        for k, (tr_idx, va_idx) in enumerate(splits):
            t0 = time.time()
            tr, va = df[tr_idx], df[va_idx]
            X_tr, y_tr, cat_idx = prep(tr, cols, "hit_2x")
            X_va, y_va, _ = prep(va, cols, "hit_2x")
            # destroy training labels — preserve class ratio
            y_shuf = rng.permutation(y_tr)
            base_rate = float(y_shuf.mean())
            ds_tr = lgb.Dataset(X_tr, label=y_shuf, categorical_feature=cat_idx, free_raw_data=False)
            ds_va = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                dict(objective="binary", metric="auc",
                     learning_rate=0.05, num_leaves=63,
                     feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                     min_data_in_leaf=200,
                     scale_pos_weight=(1 - base_rate) / max(base_rate, 1e-6),
                     verbose=-1, n_jobs=-1),
                ds_tr, num_boost_round=200,
                valid_sets=[ds_va],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            pred = booster.predict(X_va)
            auc = float(roc_auc_score(y_va, pred))
            aucs.append(auc)
            print(f"  shuffle {shuffle_idx} fold {k}: AUC={auc:.4f} t={time.time()-t0:.1f}s")

    arr = np.array(aucs)
    summary = {
        "n_shuffles": N_SHUFFLES,
        "n_folds": len(splits),
        "n_samples": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "all_aucs": arr.tolist(),
        "expected_under_h0": 0.5,
        "verdict": "clean" if abs(arr.mean() - 0.5) < 0.02 and arr.max() < 0.55 else "leakage_suspected",
    }
    (ART / "permutation_null.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
