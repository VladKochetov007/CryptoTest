"""Score all tokens with meta__lgbm and export top-N to CSV.

Usage:  python eda/score_submission.py [--top 1000] [--out scored.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts" / "meta__lgbm"
CAT_COL = "deployer_wallet_source_cex_name"


def load_features() -> tuple[pl.DataFrame, list[str]]:
    feat = pl.read_parquet(ROOT / "eda" / "features.parquet").sort("deploy_time_unix")
    meta = pl.read_parquet(ROOT / "eda" / "meta_features.parquet")
    df = feat.join(meta, on="token_id", how="left")
    cols = json.loads((ART / "summary.json").read_text())["features"]
    cols = [c for c in cols if c in df.columns]
    return df, cols


def to_pandas(df: pl.DataFrame, cols: list[str]):
    sub = df.select(cols)
    if CAT_COL in cols:
        sub = sub.with_columns(
            pl.col(CAT_COL).fill_null("__missing__").cast(pl.Categorical)
        )
    numeric = [c for c in cols if c != CAT_COL]
    sub = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    X = sub.to_pandas()
    if CAT_COL in cols:
        X[CAT_COL] = X[CAT_COL].astype("category")
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--out", default="eda/scoring/scored_submission.csv")
    args = parser.parse_args()

    df, cols = load_features()
    booster = lgb.Booster(model_file=str(ART / "model_last.txt"))
    X = to_pandas(df, cols)
    scores = booster.predict(X)

    out = df.select("token_id", "deploy_time_unix").with_columns(
        pl.Series("score", scores, dtype=pl.Float64)
    )
    if "hit_2x" in df.columns:
        out = out.with_columns(df["hit_2x"])

    idx = np.argsort(scores)[::-1][: args.top]
    top = out[idx.tolist()].sort("score", descending=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top.write_csv(out_path)
    print(f"saved {top.height} rows -> {out_path}")
    print(f"score range: {scores.min():.4f} - {scores.max():.4f}")


if __name__ == "__main__":
    main()
