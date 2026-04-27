"""Statistical EDA dump.

Strategy: build label + features as polars DataFrame once, then run univariate stat tests
per feature (continuous and categorical) against the binary target hit_2x_30min, plus
time-stability checks and pairwise rank correlation. Output: eda/eda_report.json.

No fancy charts — plots only for human-eyeballing heavy-tail / survival.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "tokens.parquet"
SLOTS = ROOT / "slot_features_60m.parquet"
ACTS = ROOT / "deployer_actions_60m.parquet"
OUT_DIR = ROOT / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS = OUT_DIR / "plots"
PLOTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def open_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tokens AS SELECT * FROM read_parquet('{TOKENS}')")
    con.execute(f"CREATE VIEW slots  AS SELECT * FROM read_parquet('{SLOTS}')")
    con.execute(f"CREATE VIEW acts   AS SELECT * FROM read_parquet('{ACTS}')")
    return con


def build_dataset(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Tokens table joined with first-slot price + 30-min peak price + early signals.

    Resulting columns include label hit_2x and instant deploy-time features only
    (no leakage into label evaluation, but post-deploy aggregates are added separately
    and clearly marked).
    """
    arrow = con.sql(
        """
        WITH first AS (
            SELECT token_id, MIN(seconds_since_deploy) AS s0
            FROM slots WHERE price_sol_per_token > 0 GROUP BY 1
        ),
        p0 AS (
            SELECT s.token_id,
                   ANY_VALUE(s.price_sol_per_token) AS p0,
                   ANY_VALUE(s.buy_volume_sol)      AS first_buy_vol_sol,
                   ANY_VALUE(s.holders_count)       AS first_holders,
                   ANY_VALUE(s.trade_count)         AS first_trade_count
            FROM slots s
            JOIN first f ON s.token_id = f.token_id AND s.seconds_since_deploy = f.s0
            WHERE s.price_sol_per_token > 0
            GROUP BY 1
        ),
        peak30 AS (
            SELECT token_id, MAX(price_sol_per_token) AS pmax_30m
            FROM slots WHERE seconds_since_deploy <= 1800 AND price_sol_per_token > 0
            GROUP BY 1
        ),
        peak60 AS (
            SELECT token_id, MAX(price_sol_per_token) AS pmax_60m
            FROM slots WHERE seconds_since_deploy <= 3599 AND price_sol_per_token > 0
            GROUP BY 1
        ),
        early60s AS (
            SELECT token_id,
                   SUM(buy_volume_sol)   AS buy_vol_60s,
                   SUM(sell_volume_sol)  AS sell_vol_60s,
                   MAX(holders_count)    AS holders_60s,
                   MAX(CASE WHEN top_wallet_bought THEN 1 ELSE 0 END) AS top_wallet_60s
            FROM slots WHERE seconds_since_deploy <= 60 GROUP BY 1
        ),
        ranked AS (
            SELECT token_id, deployer_address,
                   ROW_NUMBER() OVER (PARTITION BY deployer_address ORDER BY deploy_time_unix) - 1
                       AS deployer_prior_n
            FROM tokens
        ),
        img_clone AS (
            SELECT image_hash_sha256, COUNT(*) AS hash_seen_total
            FROM tokens WHERE image_hash_sha256 IS NOT NULL GROUP BY 1
        ),
        ticker_clash AS (
            SELECT t.token_id, COUNT(*) - 1 AS same_ticker_today
            FROM tokens t JOIN tokens t2
              ON t.ticker = t2.ticker
             AND date_trunc('day', t.deploy_time_utc) = date_trunc('day', t2.deploy_time_utc)
            GROUP BY t.token_id
        )
        SELECT
            t.token_id,
            t.deploy_time_unix,
            -- deploy-time features (allowed for Part 1)
            t.deployer_deposit_amount,
            t.deployer_wallet_balance_before,
            t.deployer_wallet_balance_after_sol,
            t.deployer_wallet_source_amount_sol,
            CASE WHEN t.deployer_wallet_source_is_cex THEN 1 ELSE 0 END AS is_cex,
            t.deployer_wallet_source_cex_name,
            CASE WHEN t.image_hash_sha256 IS NULL THEN 0 ELSE 1 END AS has_image,
            CASE WHEN t.description IS NULL OR t.description='' THEN 0 ELSE 1 END AS has_desc,
            CASE WHEN t.website IS NULL THEN 0 ELSE 1 END AS has_website,
            CASE WHEN t.twitter_handle IS NULL THEN 0 ELSE 1 END AS has_twitter,
            CASE WHEN t.telegram_link IS NULL THEN 0 ELSE 1 END AS has_telegram,
            LENGTH(t.name) AS name_len,
            LENGTH(t.ticker) AS ticker_len,
            r.deployer_prior_n,
            COALESCE(c.hash_seen_total, 1) AS image_hash_seen_total,
            COALESCE(tc.same_ticker_today, 0) AS same_ticker_today,
            -- label sources
            t.ath_market_cap_usd,
            p0.p0,
            p0.first_buy_vol_sol,
            p0.first_holders,
            p0.first_trade_count,
            peak30.pmax_30m,
            peak60.pmax_60m,
            COALESCE(e.buy_vol_60s, 0)   AS buy_vol_60s,
            COALESCE(e.sell_vol_60s, 0)  AS sell_vol_60s,
            COALESCE(e.holders_60s, 0)   AS holders_60s,
            COALESCE(e.top_wallet_60s, 0) AS top_wallet_60s
        FROM tokens t
        LEFT JOIN ranked r       USING (token_id)
        LEFT JOIN img_clone c    ON c.image_hash_sha256 = t.image_hash_sha256
        LEFT JOIN ticker_clash tc USING (token_id)
        LEFT JOIN p0             USING (token_id)
        LEFT JOIN peak30         USING (token_id)
        LEFT JOIN peak60         USING (token_id)
        LEFT JOIN early60s e     USING (token_id)
        """
    ).arrow()

    df = pl.from_arrow(arrow)
    df = df.with_columns(
        (pl.col("pmax_30m") / pl.col("p0")).alias("roi_30m"),
        (pl.col("pmax_60m") / pl.col("p0")).alias("roi_60m"),
    ).with_columns(
        (pl.col("roi_30m") >= 2.0).cast(pl.Int8).alias("hit_2x"),
        (pl.col("roi_30m") >= 5.0).cast(pl.Int8).alias("hit_5x"),
        (pl.col("ath_market_cap_usd") >= 69_000).cast(pl.Int8).alias("hit_grad"),
    )
    return df


# ---------------------------------------------------------------------------
# univariate stat tests
# ---------------------------------------------------------------------------

@dataclass
class CategoryStat:
    name: str
    n: int
    base_rate: float
    levels: list[dict]
    chi2: float
    chi2_pvalue: float
    cramer_v: float
    information_value: float


def two_proportion_z(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan")
    p_a, p_b = success_a / n_a, success_b / n_b
    p = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    if se == 0:
        return float("nan"), float("nan")
    z = (p_a - p_b) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval


def woe_iv(df: pl.DataFrame, feature: str, target: str) -> tuple[list[dict], float]:
    """Weight of evidence + Information Value per category. Smoothed by 0.5."""
    g = df.group_by(feature).agg(
        pl.len().alias("n"),
        pl.col(target).sum().alias("pos"),
    ).with_columns((pl.col("n") - pl.col("pos")).alias("neg"))
    tot_pos = g["pos"].sum()
    tot_neg = g["neg"].sum()
    rows: list[dict] = []
    iv = 0.0
    for r in g.iter_rows(named=True):
        pos = r["pos"] + 0.5
        neg = r["neg"] + 0.5
        share_pos = pos / (tot_pos + 0.5 * len(g))
        share_neg = neg / (tot_neg + 0.5 * len(g))
        woe = math.log(share_pos / share_neg)
        contrib = (share_pos - share_neg) * woe
        iv += contrib
        rows.append({
            "level": r[feature],
            "n": int(r["n"]),
            "pos": int(r["pos"]),
            "rate": float(r["pos"]) / max(1, r["n"]),
            "woe": woe,
            "iv_contrib": contrib,
        })
    return rows, iv


def chi2_cramerv(df: pl.DataFrame, feature: str, target: str) -> tuple[float, float, float]:
    ct = df.group_by([feature, target]).len().pivot(
        index=feature, on=target, values="len"
    ).fill_null(0)
    cols = [c for c in ct.columns if c != feature]
    if len(cols) < 2:
        return float("nan"), float("nan"), float("nan")
    arr = ct.select(cols).to_numpy()
    chi2, p, dof, _ = stats.chi2_contingency(arr)
    n = arr.sum()
    cramer = math.sqrt(chi2 / (n * (min(arr.shape) - 1))) if n and min(arr.shape) > 1 else float("nan")
    return float(chi2), float(p), float(cramer)


def categorical_stats(df: pl.DataFrame, feature: str, target: str = "hit_2x") -> CategoryStat:
    sub = df.drop_nulls([feature, target])
    n = sub.height
    base = float(sub[target].mean())
    levels, iv = woe_iv(sub, feature, target)
    chi2, pval, cramer = chi2_cramerv(sub, feature, target)
    return CategoryStat(
        name=feature, n=n, base_rate=base,
        levels=sorted(levels, key=lambda r: -r["n"]),
        chi2=chi2, chi2_pvalue=pval, cramer_v=cramer, information_value=iv,
    )


@dataclass
class ContinuousStat:
    name: str
    n: int
    spearman_rho: float
    spearman_p: float
    point_biserial_r: float
    point_biserial_p: float
    auc: float
    ks: float
    ks_p: float
    decile_lift: list[dict]


def auc_score(y: np.ndarray, x: np.ndarray) -> float:
    # AUC = P(x|y=1 > x|y=0). Use Mann-Whitney U formulation.
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    pos_n = int(y.sum())
    neg_n = int(len(y) - pos_n)
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average rank for ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg_rank = sums / counts
    rank_arr = avg_rank[inv]
    sum_pos = rank_arr[y == 1].sum()
    auc = (sum_pos - pos_n * (pos_n + 1) / 2) / (pos_n * neg_n)
    return float(auc)


def decile_lift(df: pl.DataFrame, feature: str, target: str) -> list[dict]:
    sub = df.drop_nulls([feature, target]).select(feature, target)
    if sub.height < 100:
        return []
    sub = sub.with_columns(
        pl.col(feature).qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True).alias("dec")
    )
    g = sub.group_by("dec").agg(
        pl.len().alias("n"),
        pl.col(target).mean().alias("rate"),
    ).sort("dec")
    base = float(sub[target].mean())
    return [
        {"decile": r["dec"], "n": int(r["n"]), "rate": float(r["rate"]),
         "lift": float(r["rate"]) / base if base else float("nan")}
        for r in g.iter_rows(named=True)
    ]


def continuous_stats(df: pl.DataFrame, feature: str, target: str = "hit_2x") -> ContinuousStat:
    sub = df.drop_nulls([feature, target])
    x = sub[feature].to_numpy().astype(float)
    y = sub[target].to_numpy().astype(int)
    n = len(x)
    rho, rho_p = stats.spearmanr(x, y)
    pb_r, pb_p = stats.pointbiserialr(y, x)
    auc = auc_score(y, x)
    ks, ks_p = stats.ks_2samp(x[y == 1], x[y == 0])
    return ContinuousStat(
        name=feature, n=n,
        spearman_rho=float(rho), spearman_p=float(rho_p),
        point_biserial_r=float(pb_r), point_biserial_p=float(pb_p),
        auc=float(auc),
        ks=float(ks), ks_p=float(ks_p),
        decile_lift=decile_lift(df, feature, target),
    )


# ---------------------------------------------------------------------------
# multivariate / time-stability
# ---------------------------------------------------------------------------

def spearman_matrix(df: pl.DataFrame, features: list[str]) -> dict:
    sub = df.select(features).drop_nulls()
    if sub.height < 100:
        return {}
    arr = sub.to_numpy().astype(float)
    rho, _ = stats.spearmanr(arr)
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    out = {}
    for i, a in enumerate(features):
        for j, b in enumerate(features):
            if j > i:
                out[f"{a}__{b}"] = float(rho[i, j])
    return out


def time_stability(df: pl.DataFrame, target: str = "hit_2x", n_chunks: int = 10) -> dict:
    sub = df.drop_nulls([target, "deploy_time_unix"]).sort("deploy_time_unix")
    if sub.height < n_chunks * 100:
        return {}
    chunks = np.array_split(sub.to_numpy(), n_chunks)  # rough; we want target column
    rates = []
    for c in chunks:
        col_idx = sub.columns.index(target)
        rates.append(float(np.mean(c[:, col_idx].astype(float))))
    return {
        "n_chunks": n_chunks,
        "rate_min": min(rates),
        "rate_max": max(rates),
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates)),
        "rates_per_chunk": rates,
    }


# ---------------------------------------------------------------------------
# extra: deployer first sell timing (rug signal)
# ---------------------------------------------------------------------------

def deployer_action_distributions(con: duckdb.DuckDBPyConnection) -> dict:
    first_sell = con.sql(
        """
        WITH d AS (
            SELECT token_id, MIN(seconds_since_deploy) sec
            FROM acts WHERE deployer_action ILIKE '%sell%' GROUP BY 1
        ) SELECT
              COUNT(*) AS n_with_dep_sell,
              approx_quantile(sec, 0.1) p10,
              approx_quantile(sec, 0.25) p25,
              approx_quantile(sec, 0.5) p50,
              approx_quantile(sec, 0.9) p90,
              AVG(CASE WHEN sec<=30 THEN 1 ELSE 0 END) le30s,
              AVG(CASE WHEN sec<=120 THEN 1 ELSE 0 END) le120s
          FROM d
        """
    ).fetchone()
    fee_claim = con.sql(
        """
        WITH d AS (
            SELECT token_id, MIN(seconds_since_deploy) sec
            FROM acts WHERE deployer_action ILIKE '%creator_fee%' GROUP BY 1
        ) SELECT COUNT(*) n, approx_quantile(sec, 0.5) p50,
                 approx_quantile(sec, 0.9) p90
          FROM d
        """
    ).fetchone()
    return {
        "first_sell": dict(zip(
            ["n", "p10", "p25", "p50", "p90", "frac_le30s", "frac_le120s"],
            first_sell)),
        "creator_fee_claim": dict(zip(["n", "p50", "p90"], fee_claim)),
    }


# ---------------------------------------------------------------------------
# bare plots — only heavy-tail confirmation, not analysis input
# ---------------------------------------------------------------------------

def make_confirmation_plots(con: duckdb.DuckDBPyConnection):
    import matplotlib.pyplot as plt
    arr = np.asarray(
        con.sql("SELECT ath_market_cap_usd FROM tokens WHERE ath_market_cap_usd>0").fetchnumpy()["ath_market_cap_usd"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    arr_s = np.sort(arr); cdf = np.arange(1, len(arr_s) + 1) / len(arr_s)
    ax.plot(arr_s, 1 - cdf)
    ax.axvline(69_000, color="red", linestyle="--", label="grad ~$69k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("ATH market cap survival (P(ATH>=x))"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "ath_survival.png", dpi=110); plt.close(fig)

    rows = con.sql(
        """
        WITH bucket AS (SELECT token_id, MAX(seconds_since_deploy) last_sec FROM slots GROUP BY 1)
        SELECT s.s, AVG(CASE WHEN bucket.last_sec >= s.s THEN 1 ELSE 0 END)
        FROM bucket, generate_series(0, 3600, 30) s(s) GROUP BY s.s ORDER BY s.s
        """
    ).fetchall()
    sec = np.array([r[0] for r in rows]) / 60
    alive = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sec, alive)
    ax.set_xlabel("min since deploy"); ax.set_ylabel("frac trading >= t")
    ax.set_title("Token activity survival in first 60min")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "survival.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# top-level runner
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "is_cex",
    "has_image",
    "has_desc",
    "has_website",
    "has_twitter",
    "has_telegram",
    "deployer_wallet_source_cex_name",
]

CONTINUOUS_FEATURES = [
    "deployer_deposit_amount",
    "deployer_wallet_balance_before",
    "deployer_wallet_balance_after_sol",
    "deployer_wallet_source_amount_sol",
    "deployer_prior_n",
    "image_hash_seen_total",
    "same_ticker_today",
    "name_len",
    "ticker_len",
]

POSTBUY_CONTINUOUS = [
    "first_buy_vol_sol",
    "first_holders",
    "first_trade_count",
    "buy_vol_60s",
    "sell_vol_60s",
    "holders_60s",
    "top_wallet_60s",
]


def main():
    con = open_con()

    df = build_dataset(con)
    df.write_parquet(OUT_DIR / "feature_table.parquet")
    n_total = df.height
    n_with_label = df.drop_nulls(["hit_2x"]).height

    base_rates = {
        "n_total": int(n_total),
        "n_with_first_slot_price": int(n_with_label),
        "hit_2x_30m": float(df["hit_2x"].mean()),
        "hit_5x_30m": float(df["hit_5x"].mean()),
        "hit_grad_69k": float(df["hit_grad"].mean()),
        "roi_30m_p50": float(df["roi_30m"].quantile(0.5) or 0),
        "roi_30m_p90": float(df["roi_30m"].quantile(0.9) or 0),
        "roi_30m_p99": float(df["roi_30m"].quantile(0.99) or 0),
    }

    cat_stats = {f: categorical_stats(df, f).__dict__ for f in CATEGORICAL_FEATURES}
    cont_stats = {f: continuous_stats(df, f).__dict__ for f in CONTINUOUS_FEATURES}
    postbuy_stats = {f: continuous_stats(df, f).__dict__ for f in POSTBUY_CONTINUOUS}

    rank_corr = spearman_matrix(df, CONTINUOUS_FEATURES)

    stability = time_stability(df, "hit_2x", n_chunks=10)
    stability_grad = time_stability(df, "hit_grad", n_chunks=10)

    actions = deployer_action_distributions(con)

    make_confirmation_plots(con)

    report = {
        "base_rates": base_rates,
        "categorical_features": cat_stats,
        "continuous_features": cont_stats,
        "postbuy_continuous_features (POST-BUY, NOT for Part1)": postbuy_stats,
        "spearman_pairwise_continuous": rank_corr,
        "time_stability_hit_2x": stability,
        "time_stability_hit_grad": stability_grad,
        "deployer_action_distributions": actions,
    }
    out_path = OUT_DIR / "eda_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
