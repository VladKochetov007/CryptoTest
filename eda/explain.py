"""SHAP feature importance for trained models + 0-100 score calibration + 1k token output.

Loads the LAST fold's LGBM model (trained on the longest expanding window) plus the
out-of-fold predictions, calibrates probabilities with isotonic regression on the OOF set,
maps to 0-100 score, and emits a buy/skip column at threshold tuned for ROI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
ART = ROOT / "eda" / "artifacts"
OUT = ROOT / "eda" / "scoring"
OUT.mkdir(exist_ok=True)


def load_oof(feature_set: str, algo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids = np.load(ART / f"{feature_set}__token_ids.npy")
    y = np.load(ART / f"{feature_set}__y.npy")
    oof = np.load(ART / f"{feature_set}__{algo}" / "oof_pred.npy")
    mask = ~np.isnan(oof)
    return token_ids[mask], y[mask], oof[mask]


def shap_feature_importance(feature_set: str, model_features: list[str], n_sample: int = 20_000):
    """Tree-SHAP on a held-out sample using the model trained on the LAST fold."""
    import lightgbm as lgb
    import shap
    booster = lgb.Booster(model_file=str(ART / f"{feature_set}__lgbm" / "model_last.txt"))
    df = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    cols = [c for c in model_features if c in df.columns]
    sample = df.tail(n_sample).select(cols).with_columns(
        pl.col("deployer_wallet_source_cex_name").fill_null("__missing__").cast(pl.Categorical)
        if "deployer_wallet_source_cex_name" in cols else pl.lit(None)
    ).to_pandas()
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    importance = np.abs(shap_vals).mean(axis=0)
    ranking = sorted(zip(cols, importance), key=lambda kv: -kv[1])
    return ranking, shap_vals, sample


def calibrate(y: np.ndarray, p: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def threshold_sweep(y: np.ndarray, p: np.ndarray) -> dict:
    """Sweep thresholds; report per-bin precision, lift, expected fraction selected."""
    base = float(y.mean())
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        sel = p >= t
        n = int(sel.sum())
        if n == 0:
            continue
        precision = float(y[sel].mean())
        rows.append({
            "threshold": float(t),
            "n_selected": n,
            "selection_rate": n / len(p),
            "precision": precision,
            "lift": precision / base,
        })
    return {"base_rate": base, "rows": rows}


def main():
    summaries = {}
    for fs in ("instant", "with60s"):
        per_algo = {}
        for algo in ("lgbm", "xgb", "catboost"):
            path = ART / f"{fs}__{algo}"
            if not (path / "summary.json").exists():
                continue
            s = json.loads((path / "summary.json").read_text())
            tids, y, p = load_oof(fs, algo)
            auc = roc_auc_score(y, p)
            pr = average_precision_score(y, p)
            br = brier_score_loss(y, p)
            per_algo[algo] = {"auc": auc, "pr_auc": pr, "brier": br,
                               "fold_auc": [r["auc"] for r in s["fold_results"]]}
        summaries[fs] = per_algo

    (OUT / "model_comparison.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))

    # ---- pick best instant-model for output ----
    best_fs = "instant"
    best_algo = max(summaries[best_fs], key=lambda a: summaries[best_fs][a]["auc"])
    print(f"\n[best] {best_fs} / {best_algo} AUC={summaries[best_fs][best_algo]['auc']:.4f}")

    tids, y, p = load_oof(best_fs, best_algo)
    iso = calibrate(y, p)
    p_cal = iso.predict(p)

    sweep = threshold_sweep(y, p_cal)
    (OUT / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    # SHAP on best LGBM
    if best_algo == "lgbm":
        s = json.loads((ART / f"{best_fs}__lgbm" / "summary.json").read_text())
        ranking, shap_vals, sample = shap_feature_importance(best_fs, s["features"])
        (OUT / "shap_ranking.json").write_text(json.dumps(
            [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranking], indent=2))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap
            shap.summary_plot(shap_vals, sample, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(OUT / "shap_summary.png", dpi=120)
            plt.close()
        except Exception as e:
            print("plot failed:", e)

    # ---- final scoring table for tokens with OOF predictions ----
    base_rate = float(y.mean())
    tbl = pl.DataFrame({"token_id": tids, "y": y, "p_raw": p, "p_cal": p_cal})
    score = np.clip(np.round(p_cal * 100, 1), 0, 100)
    tbl = tbl.with_columns(pl.Series("score_0_100", score))
    # decision threshold = max-lift threshold with selection_rate >= 5%.
    # The literal max-lift threshold (~0.99) selects almost nothing — useful as
    # the "high-conviction" tier, but a real bot needs a softer "buy at all"
    # threshold to fund position sizing.
    rows = sweep["rows"]
    soft = [r for r in rows if r["selection_rate"] >= 0.05]
    high = [r for r in rows if r["selection_rate"] >= 0.001]
    best_soft = max(soft, key=lambda r: r["lift"]) if soft else rows[-1]
    best_high = max(high, key=lambda r: r["lift"]) if high else rows[-1]
    tbl = tbl.with_columns(
        (pl.col("p_cal") >= best_soft["threshold"]).alias("buy_decision"),
        (pl.col("p_cal") >= best_high["threshold"]).alias("high_conviction"),
    )
    tbl.write_parquet(OUT / "scored_tokens.parquet")

    # 1k token sample (chronological tail = most recent buys)
    full = (
        pl.read_parquet(FEAT)
        .select("token_id", "deploy_time_unix", "deployer_address",
                "name_len", "ticker_len", "deployer_deposit_amount", "is_cex",
                "deployer_wallet_source_cex_name", "ath_market_cap_usd", "hit_2x")
        .join(tbl, on="token_id", how="inner")
        .sort("deploy_time_unix", descending=True)
        .head(1000)
    )
    full.write_csv(OUT / "scored_1000_recent.csv")
    full.write_parquet(OUT / "scored_1000_recent.parquet")
    summary = {
        "best_model": f"{best_fs}/{best_algo}",
        "auc": summaries[best_fs][best_algo]["auc"],
        "pr_auc": summaries[best_fs][best_algo]["pr_auc"],
        "brier": summaries[best_fs][best_algo]["brier"],
        "base_rate": base_rate,
        "buy_decision": {
            "threshold": best_soft["threshold"], "lift": best_soft["lift"],
            "precision": best_soft["precision"], "n_selected": best_soft["n_selected"],
            "selection_rate": best_soft["selection_rate"],
        },
        "high_conviction": {
            "threshold": best_high["threshold"], "lift": best_high["lift"],
            "precision": best_high["precision"], "n_selected": best_high["n_selected"],
            "selection_rate": best_high["selection_rate"],
        },
    }
    (OUT / "scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
