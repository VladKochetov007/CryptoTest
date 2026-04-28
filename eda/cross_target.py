"""Cross-target experiment: train on hit_Nx, evaluate on hit_Mx.

Hypothesis: training on a harder target (hit_5x, hit_10x) may improve lift at the
top of the score distribution for the softer target (hit_2x) with less risk,
because the model learns to identify only the truly exceptional launches.

Outputs:
- eda/cross_target/auc_matrix.json     — 4x4 OOF AUC cross-eval
- eda/cross_target/lift_matrix.json    — precision at top decile
- eda/cross_target/winrate_matrix.json — win rate at top decile
- eda/cross_target/summary.md          — human-readable table
- eda/plots/cross_target_heatmap.png   — visual for management
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
OUT_DIR = ROOT / "eda" / "cross_target"
PLOT_DIR = ROOT / "eda" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEPLOY_TIME_FEATURES = [
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
CAT_COL = "deployer_wallet_source_cex_name"

TARGETS = ["hit_2x", "hit_3x", "hit_5x", "hit_10x"]


def build_labels(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("roi_30m") >= 3.0).cast(pl.Int8).alias("hit_3x"),
        (pl.col("roi_30m") >= 10.0).cast(pl.Int8).alias("hit_10x"),
    )


def walkforward_splits(n: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = np.linspace(0, n, n_folds + 2, dtype=int)
    splits = []
    for k in range(n_folds):
        tr_end = edges[k + 1]
        va_start, va_end = edges[k + 1], edges[k + 2]
        if tr_end <= 0 or va_end - va_start <= 0:
            continue
        splits.append((np.arange(0, tr_end), np.arange(va_start, va_end)))
    return splits


def prep_X(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    sub = df.select(cols)
    for c in cols:
        if c == CAT_COL:
            sub = sub.with_columns(
                pl.col(c).fill_null("__missing__").cast(pl.Categorical).to_physical().alias(c)
            )
        else:
            sub = sub.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return sub.to_pandas()


def train_lgbm_oof(df: pl.DataFrame, cols: list[str], target: str) -> np.ndarray:
    """Walk-forward LGBM OOF predictions for a given target label."""
    n = df.height
    splits = walkforward_splits(n)
    oof = np.full(n, np.nan)
    cat_idx = [cols.index(CAT_COL)] if CAT_COL in cols else []

    for tr_idx, va_idx in splits:
        tr, va = df[tr_idx], df[va_idx]
        # skip fold if target is nearly all one class (degenerate)
        base = float(tr[target].mean())
        if base < 0.001 or base > 0.999:
            continue

        X_tr = prep_X(tr, cols)
        y_tr = tr[target].to_numpy().astype(int)
        X_va = prep_X(va, cols)

        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=va[target].to_numpy().astype(int), free_raw_data=False)

        params = dict(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            min_data_in_leaf=200,
            scale_pos_weight=(1 - base) / max(base, 1e-6),
            verbose=-1, n_jobs=-1,
        )
        booster = lgb.train(
            params, ds_tr, num_boost_round=600,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(X_va)

    return oof


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    top_frac: float = 0.10,
) -> dict[str, float]:
    """AUC + precision and positive rate at top-decile selection."""
    mask = ~np.isnan(scores) & ~np.isnan(y_true.astype(float))
    y, s = y_true[mask], scores[mask]
    if y.sum() == 0 or len(y) < 10:
        return {"auc": float("nan"), "precision_top": float("nan"), "base_rate": float("nan")}

    auc = float(roc_auc_score(y, s))
    threshold = np.quantile(s, 1 - top_frac)
    selected = y[s >= threshold]
    base = float(y.mean())
    precision = float(selected.mean()) if len(selected) > 0 else float("nan")
    lift = (precision / base) if base > 0 else float("nan")
    return {"auc": auc, "precision_top": precision, "lift_top": lift, "base_rate": base}


def plot_heatmap(matrix: dict, title: str, fmt: str, path: Path):
    values = np.array([[matrix[tr][ev] for ev in TARGETS] for tr in TARGETS], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(values, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_yticks(range(len(TARGETS)))
    ax.set_xticklabels([t.replace("hit_", "") for t in TARGETS])
    ax.set_yticklabels([t.replace("hit_", "") for t in TARGETS])
    ax.set_xlabel("Eval target")
    ax.set_ylabel("Train target")
    ax.set_title(title)
    for i in range(len(TARGETS)):
        for j in range(len(TARGETS)):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                        color="black" if 0.2 < (v - np.nanmin(values)) / max(np.nanmax(values) - np.nanmin(values), 1e-6) < 0.8 else "white")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def main():
    print("[cross_target] loading features")
    df = pl.read_parquet(FEAT)
    df = build_labels(df)

    # keep only rows with p0 (label-eligible)
    df = df.drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    cols = [c for c in DEPLOY_TIME_FEATURES if c in df.columns]
    print(f"  rows: {df.height}, cols: {len(cols)}")

    base_rates = {t: float(df[t].mean()) for t in TARGETS}
    print("  base rates:", {t: f"{r:.3f}" for t, r in base_rates.items()})

    oof_scores: dict[str, np.ndarray] = {}
    for train_target in TARGETS:
        print(f"[cross_target] training on {train_target} (base={base_rates[train_target]:.3f})")
        oof = train_lgbm_oof(df, cols, train_target)
        oof_scores[train_target] = oof
        valid_frac = (~np.isnan(oof)).mean()
        print(f"  OOF coverage: {valid_frac:.3f}")

    auc_mat: dict[str, dict[str, float]] = {}
    lift_mat: dict[str, dict[str, float]] = {}
    prec_mat: dict[str, dict[str, float]] = {}

    for train_target in TARGETS:
        auc_mat[train_target] = {}
        lift_mat[train_target] = {}
        prec_mat[train_target] = {}
        for eval_target in TARGETS:
            y_eval = df[eval_target].to_numpy().astype(float)
            m = compute_metrics(y_eval, oof_scores[train_target], top_frac=0.10)
            auc_mat[train_target][eval_target] = m["auc"]
            lift_mat[train_target][eval_target] = m.get("lift_top", float("nan"))
            prec_mat[train_target][eval_target] = m.get("precision_top", float("nan"))

    (OUT_DIR / "auc_matrix.json").write_text(json.dumps(auc_mat, indent=2))
    (OUT_DIR / "lift_matrix.json").write_text(json.dumps(lift_mat, indent=2))
    (OUT_DIR / "precision_matrix.json").write_text(json.dumps(prec_mat, indent=2))

    print("\n=== AUC Matrix (rows=train_target, cols=eval_target) ===")
    header = f"{'':12}" + "".join(f"{t:14}" for t in TARGETS)
    print(header)
    for tr in TARGETS:
        row = f"{tr:12}" + "".join(f"{auc_mat[tr][ev]:14.4f}" for ev in TARGETS)
        print(row)

    print("\n=== Lift@top-10% Matrix ===")
    print(header)
    for tr in TARGETS:
        row = f"{tr:12}" + "".join(f"{lift_mat[tr][ev]:14.3f}" for ev in TARGETS)
        print(row)

    plot_heatmap(auc_mat, "AUC cross-target matrix", "{:.3f}", PLOT_DIR / "cross_target_auc.png")
    plot_heatmap(lift_mat, "Lift@top-10% cross-target matrix", "{:.2f}", PLOT_DIR / "cross_target_lift.png")

    summary_lines = [
        "# Cross-Target Results\n",
        f"Train/eval on {TARGETS}. Top-10% selection.\n",
        "\n## AUC Matrix\n",
        "| Train \\ Eval | " + " | ".join(t.replace("hit_", "") for t in TARGETS) + " |",
        "|" + "--|" * (len(TARGETS) + 1),
    ]
    for tr in TARGETS:
        row = f"| {tr.replace('hit_', '')} | " + " | ".join(f"{auc_mat[tr][ev]:.4f}" for ev in TARGETS) + " |"
        summary_lines.append(row)

    summary_lines += [
        "\n## Lift@top-10% Matrix\n",
        "| Train \\ Eval | " + " | ".join(t.replace("hit_", "") for t in TARGETS) + " |",
        "|" + "--|" * (len(TARGETS) + 1),
    ]
    for tr in TARGETS:
        row = f"| {tr.replace('hit_', '')} | " + " | ".join(f"{lift_mat[tr][ev]:.3f}" for ev in TARGETS) + " |"
        summary_lines.append(row)

    summary_lines += ["\n## Base Rates\n"]
    for t, r in base_rates.items():
        summary_lines.append(f"- {t}: {r:.4f} ({r*100:.2f}%)")

    summary = "\n".join(summary_lines)
    (OUT_DIR / "summary.md").write_text(summary)
    print(f"\n[cross_target] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
