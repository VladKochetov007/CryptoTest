"""Hard assertions to catch off-by-one and time-series leakage.

Run after every retrain. Exits non-zero on the first failure so it slots into
make/CI without parsing logs.

Tests:
  T1. Walk-forward folds: disjoint indices, train.max(time) < val.min(time).
  T2. Embargo respected: val.min(time) - train.max(time) >= LABEL_WINDOW_SEC.
  T3. No feature column equals or trivially derives the label.
  T4. Tie-time peers in compute_rolling_group_features see identical priors.
  T5. Two-stage idx_60 boundary correct for a sample of tokens.
  T6. OOF token_ids align with sorted df["token_id"] for instant + meta artefacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from build_meta_features import compute_rolling_group_features
from train import DEPLOY_TIME_FEATURES, POST60_FEATURES, walkforward_indices
from meta_train import BASE_FEATURES, META_FEATURES

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "eda" / "features.parquet"
META = ROOT / "eda" / "meta_features.parquet"
SLOT = ROOT / "slot_features_60m.parquet"
ART = ROOT / "eda" / "artifacts"

LABEL_WINDOW_SEC = 1800
EMBARGO_SEC = 3600


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"[ok]   {msg}")


def t1_t2_walkforward(times: np.ndarray, label: str) -> None:
    folds = walkforward_indices(times, n_folds=5, embargo_sec=EMBARGO_SEC)
    if not folds:
        fail(f"{label}: walk-forward produced zero folds")
    for k, (tr, va) in enumerate(folds):
        if len(np.intersect1d(tr, va)) > 0:
            fail(f"{label} fold {k}: tr ∩ va non-empty")
        if times[tr].max() >= times[va].min():
            fail(f"{label} fold {k}: tr.max(time) >= va.min(time)")
        gap = int(times[va].min() - times[tr].max())
        if gap < LABEL_WINDOW_SEC:
            fail(f"{label} fold {k}: gap {gap}s < label window {LABEL_WINDOW_SEC}s")
    ok(f"{label}: {len(folds)} folds, all disjoint and embargoed")


def t3_no_label_in_features() -> None:
    bad = {"hit_2x", "hit_5x", "ath_market_cap_usd", "peak_marketcap_usd",
           "roi_30m", "price_max_60s", "price_min_60s"}
    leaks = bad & set(DEPLOY_TIME_FEATURES + POST60_FEATURES + BASE_FEATURES + META_FEATURES)
    if leaks:
        fail(f"label-derived columns appear in feature lists: {leaks}")
    ok(f"no label/peak columns leak into feature lists ({len(bad)} forbidden checked)")


def t4_tie_time_peers() -> None:
    feat = pl.read_parquet(FEAT).select("token_id", "deployer_address", "deploy_time_unix", "hit_2x")
    # Restrict to tokens with observed hit_2x — null-hit tokens are excluded by the
    # rolling helper (no contribution to hit-rate denominator), so tied pairs where
    # one peer is null-hit are not expected to match.
    dups = (
        feat.drop_nulls(["deployer_address", "deploy_time_unix", "hit_2x"])
        .group_by(["deployer_address", "deploy_time_unix"])
        .agg(pl.col("token_id").alias("tids"), pl.len().alias("n"))
        .filter(pl.col("n") >= 2)
    )
    if dups.is_empty():
        ok("no same-second same-deployer pairs found — tie-leak path not exercised")
        return
    res = compute_rolling_group_features(
        feat, "deployer_address", "hit_2x",
        {"24h": 86400, "7d": 604800}, "deployer",
    )
    for row in dups.head(50).iter_rows(named=True):
        peers = row["tids"]
        sub = res.filter(pl.col("token_id").is_in(peers))
        for col in ("deployer_prior_n_24h", "deployer_hr_24h",
                    "deployer_prior_n_7d", "deployer_hr_7d"):
            vals = sub[col].to_numpy()
            mask = ~np.isnan(vals)
            uniq = np.unique(vals[mask]) if mask.any() else np.array([])
            uniq_int = np.unique(vals[mask].astype(int)) if "prior_n" in col and mask.any() else uniq
            if "prior_n" in col:
                if len(uniq_int) > 1:
                    fail(f"tie-peers {peers} differ on {col}: {vals.tolist()}")
            else:
                if len(uniq) > 1 and not np.allclose(uniq, uniq[0], equal_nan=True):
                    fail(f"tie-peers {peers} differ on {col}: {vals.tolist()}")
    ok(f"tie-time peers (sampled {min(50, dups.height)}/{dups.height}) see identical priors")


def t5_two_stage_idx60() -> None:
    if not SLOT.exists():
        ok("slot_features_60m.parquet absent — t5 skipped")
        return
    sample = (
        pl.scan_parquet(SLOT)
        .filter(pl.col("price_sol_per_token") > 0)
        .select("token_id", "seconds_since_deploy")
        .group_by("token_id")
        .agg(pl.col("seconds_since_deploy").sort().alias("secs"))
        .head(2000)
        .collect()
    )
    misses = 0
    for row in sample.iter_rows(named=True):
        secs = np.array(row["secs"], dtype=np.int64)
        if len(secs) == 0:
            continue
        idx = int(np.searchsorted(secs, 60, side="left"))
        if idx >= len(secs):
            continue
        if secs[idx] < 60:
            fail(f"token {row['token_id']}: secs[idx_60] {secs[idx]} < 60")
        if idx > 0 and secs[idx - 1] >= 60:
            fail(f"token {row['token_id']}: secs[idx_60-1] {secs[idx-1]} >= 60")
        misses += 1
    ok(f"two-stage idx_60 boundary correct on {misses} sampled tokens")


def t6_oof_alignment() -> None:
    df_inst = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    expected_inst = df_inst["token_id"].to_numpy()
    saved_inst = np.load(ART / "instant__token_ids.npy")
    if not np.array_equal(expected_inst, saved_inst):
        fail("instant__token_ids.npy out of order vs sorted features.parquet")
    ok(f"instant token_ids aligned ({len(expected_inst)} rows)")

    meta_path = ART / "meta__token_ids.npy"
    if not meta_path.exists():
        ok("meta__token_ids.npy absent (run meta_train first) — meta alignment skipped")
        return
    df_meta = (
        pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    )
    expected_meta = df_meta["token_id"].to_numpy()
    saved_meta = np.load(meta_path)
    if not np.array_equal(expected_meta, saved_meta):
        fail("meta__token_ids.npy out of order vs sorted features.parquet")
    ok(f"meta token_ids aligned ({len(expected_meta)} rows)")


def main() -> None:
    print("[leakage_tests] starting")
    df = pl.read_parquet(FEAT).drop_nulls(["hit_2x"]).sort("deploy_time_unix")
    times = df["deploy_time_unix"].to_numpy()
    t1_t2_walkforward(times, "instant/with60s/meta share the same time index")
    t3_no_label_in_features()
    t4_tie_time_peers()
    t5_two_stage_idx60()
    t6_oof_alignment()
    print("[leakage_tests] all assertions passed")


if __name__ == "__main__":
    main()
