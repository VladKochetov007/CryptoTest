"""SHAP beeswarm plot for meta__lgbm — loads model, computes SHAP on 2k OOF sample."""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts" / "meta__lgbm"
PLOT_DIR = ROOT / "eda" / "plots"

N_SAMPLE = 2000
TOP_N = 30


def load_full_feature_matrix() -> pl.DataFrame:
    feat = pl.read_parquet(ROOT / "eda" / "features.parquet")
    meta = pl.read_parquet(ROOT / "eda" / "meta_features.parquet")
    return feat.join(meta, on="token_id", how="inner")


def main():
    shap_records = json.loads((ART / "shap_meta.json").read_text())
    model_features = [r["feature"] for r in shap_records]

    booster = lgb.Booster(model_file=str(ART / "model_last.txt"))

    df = load_full_feature_matrix()
    df = df.drop_nulls("hit_2x")

    rng = np.random.default_rng(42)
    idx = rng.choice(df.height, min(N_SAMPLE, df.height), replace=False)
    sample = df[idx]

    cat_col = "deployer_wallet_source_cex_name"
    sample = sample.with_columns(
        pl.col(cat_col).fill_null("__missing__").cast(pl.Categorical)
    )
    num_cols = [c for c in model_features if c != cat_col]
    sample = sample.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols])

    X = sample.select(model_features).to_pandas()
    X[cat_col] = X[cat_col].astype("category")

    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X)

    # Top-N features by mean |SHAP| on this sample
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:TOP_N]
    top_features = [model_features[i] for i in top_idx]

    shap_top = shap_vals[:, top_idx]
    X_top = X[top_features]

    fig, ax = plt.subplots(figsize=(10, 11))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=top_features,
        max_display=TOP_N,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    ax = plt.gca()
    summary = json.loads((ART / "summary.json").read_text())
    oof_auc = summary.get("oof_auc", 0)
    ax.set_title(f"meta__lgbm SHAP beeswarm — top {TOP_N} (n={N_SAMPLE}, AUC {oof_auc:.4f})", fontsize=11)
    plt.tight_layout()
    out = PLOT_DIR / "shap_beeswarm_top30.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
