"""Generate diagnostic plots for meta-features after build_meta_features.py completes.

Plots saved to eda/plots/:
1. meta_feature_iv_bar.png       — IV by feature (sorted)
2. meta_handle_len_hit_rate.png  — hit rate by handle_len bucket
3. meta_meme_kw_hit_rate.png     — hit rate: meme kw vs not, by hour
4. meta_deployer_scale_auc.png   — univariate AUC at 1h/6h/24h/7d windows
5. meta_desc_template_hit.png    — hit_2x rate by desc template score
6. meta_timescale_correlation.png — correlation matrix of multi-scale features
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
META = ROOT / "eda" / "meta_features.parquet"
PLOT_DIR = ROOT / "eda" / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def information_value(y: np.ndarray, x: np.ndarray, bins: int = 10) -> float:
    """Binary IV for a numeric feature."""
    x = x.astype(float)
    mask = np.isfinite(x)
    if mask.sum() < 50 or np.unique(x[mask]).size < 2:
        return 0.0
    x_clean = x[mask]
    y_clean = y[mask].astype(int)
    try:
        x_q = pl.Series(x_clean).qcut(bins, allow_duplicates=True).cast(pl.String)
    except Exception:
        return 0.0
    df = pl.DataFrame({"y": y_clean, "bin": x_q})
    agg = df.group_by("bin").agg(
        pl.col("y").sum().alias("events"),
        (pl.len() - pl.col("y").sum()).alias("non_events"),
    )
    total_e = agg["events"].sum()
    total_n = agg["non_events"].sum()
    if total_e == 0 or total_n == 0:
        return 0.0
    iv = 0.0
    for row in agg.iter_rows(named=True):
        pct_e = max(row["events"] / total_e, 1e-8)
        pct_n = max(row["non_events"] / total_n, 1e-8)
        woe = np.log(pct_e / pct_n)
        iv += (pct_e - pct_n) * woe
    return float(iv)


def univar_auc(y: np.ndarray, x: np.ndarray) -> float:
    mask = ~np.isnan(x.astype(float))
    if mask.sum() < 100 or y[mask].sum() == 0:
        return 0.5
    try:
        return float(roc_auc_score(y[mask].astype(int), x[mask].astype(float)))
    except Exception:
        return 0.5


def plot_iv_bar(df: pl.DataFrame, meta_cols: list[str], y: np.ndarray):
    ivs = {}
    for c in meta_cols:
        if c not in df.columns:
            continue
        x = df[c].to_numpy().astype(float)
        ivs[c] = information_value(y, x)

    ivs_sorted = sorted(ivs.items(), key=lambda x: -x[1])
    names = [n for n, _ in ivs_sorted]
    vals = [v for _, v in ivs_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0.02 else "#95a5a6" for v in vals]
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.axvline(0.02, color="orange", linestyle="--", linewidth=1, label="IV=0.02 threshold")
    ax.set_xlabel("Information Value")
    ax.set_title("Meta-feature Information Value (IV > 0.02 = useful)")
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "meta_feature_iv_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
    return ivs_sorted


def plot_deployer_scale_auc(df: pl.DataFrame, y: np.ndarray):
    scales = {
        "all-time (existing)": "deployer_prior_n",
        "7d": "deployer_prior_n_7d",
        "24h": "deployer_prior_n_24h",
        "6h": "deployer_prior_n_6h",
        "1h": "deployer_prior_n_1h",
    }
    hr_scales = {
        "7d hit_rate": "deployer_hr_7d",
        "24h hit_rate": "deployer_hr_24h",
        "1h hit_rate": "deployer_hr_1h",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    all_scales = list(scales.items()) + list(hr_scales.items())
    aucs = [univar_auc(y, df[c].to_numpy().astype(float)) if c in df.columns else 0.5
            for _, c in all_scales]
    names = [n for n, _ in all_scales]

    ax = axes[0]
    bars = ax.bar(range(len(names)), aucs, color=["#3498db"] * len(scales) + ["#e74c3c"] * len(hr_scales))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Univariate AUC")
    ax.set_title("Deployer count/hitrate at different time scales")
    ax.set_ylim(0.48, 0.75)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7)

    # Funder scales
    funder_scales = {
        "all-time (existing)": "funder_prior_n",
        "24h": "funder_prior_n_24h",
        "24h hit_rate": "funder_hr_24h",
        "existing hit_rate": "funder_prior_hit20k",
    }
    faucs = [univar_auc(y, df[c].to_numpy().astype(float)) if c in df.columns else 0.5
             for c in funder_scales.values()]
    ax2 = axes[1]
    ax2.bar(range(len(funder_scales)), faucs, color=["#2ecc71"] * len(funder_scales))
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(funder_scales)))
    ax2.set_xticklabels(list(funder_scales.keys()), rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Univariate AUC")
    ax2.set_title("Funder features at different time scales")
    ax2.set_ylim(0.48, 0.75)
    for i, v in enumerate(faucs):
        ax2.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    path = PLOT_DIR / "meta_deployer_scale_auc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_hit_rate_by_group(y: np.ndarray, x: np.ndarray, label: str, path: Path, n_bins: int = 5):
    df = pl.DataFrame({"y": y.astype(int), "x": x.astype(float)}).drop_nulls()
    df = df.filter(pl.col("x").is_finite())
    try:
        df = df.with_columns(
            pl.col("x").qcut(n_bins, allow_duplicates=True).cast(pl.String).alias("bin")
        )
    except Exception:
        return
    agg = df.group_by("bin").agg(
        pl.col("y").mean().alias("hit_rate"),
        pl.len().alias("count"),
    ).sort("bin")

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(len(agg))
    ax.bar(xs, agg["hit_rate"].to_list(), color="#3498db", alpha=0.8)
    ax.axhline(float(y.mean()), color="orange", linestyle="--", label=f"overall {y.mean():.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(agg["bin"].to_list(), rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("hit_2x rate")
    ax.set_title(f"hit_2x rate by {label}")
    ax.legend()
    # add count labels
    for i, (hr, cnt) in enumerate(zip(agg["hit_rate"].to_list(), agg["count"].to_list())):
        ax.text(i, hr + 0.002, f"n={cnt}", ha="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_meme_kw_by_hour(df: pl.DataFrame, y: np.ndarray):
    """hit_2x rate for meme vs non-meme tokens by UTC hour."""
    if "name_has_meme_kw" not in df.columns or "utc_hour" not in df.columns:
        return
    combo = pl.DataFrame({
        "y": y.astype(int),
        "kw": df["name_has_meme_kw"].to_numpy().astype(int),
        "hour": df["utc_hour"].to_numpy().astype(int),
    })
    agg = combo.group_by(["hour", "kw"]).agg(pl.col("y").mean().alias("hr"), pl.len().alias("n")).sort("hour")

    fig, ax = plt.subplots(figsize=(12, 4))
    for kw_val, label, color in [(0, "no meme kw", "#3498db"), (1, "meme kw", "#e74c3c")]:
        sub = agg.filter(pl.col("kw") == kw_val)
        ax.plot(sub["hour"].to_list(), sub["hr"].to_list(), marker="o", label=label, color=color, linewidth=2)
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("hit_2x rate")
    ax.set_title("hit_2x rate by UTC hour: meme keyword vs not")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "meta_meme_kw_by_hour.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_text_auc_comparison(df: pl.DataFrame, y: np.ndarray):
    """Quick bar chart: univariate AUC for all text features."""
    text_cols = [
        "handle_len", "handle_digit_ratio", "handle_has_underscore",
        "handle_ends_in_digits", "handle_is_celeb", "handle_contains_ticker",
        "handle_prior_n_24h", "handle_hr_24h",
        "desc_has_url", "desc_has_deployed_template", "desc_exclamation_count",
        "desc_template_score", "desc_word_count", "desc_has_address",
        "name_has_meme_kw", "ticker_has_meme_kw", "name_word_count",
        "name_digit_count", "ticker_digit_count", "ticker_special_count",
        "ticker_len_4_5", "meme_kw_hr_24h",
    ]
    aucs = []
    names = []
    for c in text_cols:
        if c not in df.columns:
            continue
        auc = univar_auc(y, df[c].to_numpy().astype(float))
        aucs.append(auc)
        names.append(c)

    order = np.argsort(aucs)[::-1]
    aucs_s = [aucs[i] for i in order]
    names_s = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if a > 0.52 else "#95a5a6" for a in aucs_s]
    ax.barh(range(len(names_s)), aucs_s, color=colors)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--")
    ax.axvline(0.52, color="orange", linestyle="--", label="AUC 0.52 threshold")
    ax.set_xlabel("Univariate AUC (vs hit_2x)")
    ax.set_title("Text/meta feature univariate AUC")
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "meta_text_auc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def main():
    print("[meta_eda] loading data")
    base = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    meta = pl.read_parquet(META)
    df = base.join(meta, on="token_id", how="left")
    y = df["hit_2x"].to_numpy().astype(int)
    print(f"  rows: {df.height}, base_rate: {y.mean():.4f}")

    meta_cols = [c for c in meta.columns if c != "token_id"]

    print("[meta_eda] plotting IV bar")
    ivs = plot_iv_bar(df, meta_cols, y)

    print("[meta_eda] top-10 IV features:")
    for name, iv in ivs[:10]:
        print(f"  IV={iv:.4f}  {name}")

    print("[meta_eda] deployer time-scale AUC comparison")
    plot_deployer_scale_auc(df, y)

    print("[meta_eda] hit rate by handle_len")
    if "handle_len" in df.columns:
        plot_hit_rate_by_group(y, df["handle_len"].to_numpy().astype(float),
                               "handle_len", PLOT_DIR / "meta_handle_len_hitrate.png")

    print("[meta_eda] hit rate by desc template score")
    if "desc_template_score" in df.columns:
        plot_hit_rate_by_group(y, df["desc_template_score"].to_numpy().astype(float),
                               "desc_template_score", PLOT_DIR / "meta_desc_template_hitrate.png")

    print("[meta_eda] meme keyword by hour")
    plot_meme_kw_by_hour(df, y)

    print("[meta_eda] text feature AUC comparison")
    plot_text_auc_comparison(df, y)

    print(f"[meta_eda] all plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
