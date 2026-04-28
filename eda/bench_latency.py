"""Measure single-token inference latency: rolling lookup → features → LGBM → score.

Run from repo root:
    .venv/bin/python eda/bench_latency.py

Requires eda/artifacts/instant__lgbm/model_last.txt (from train.py).
"""
import re
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

model = lgb.Booster(model_file=str(ROOT / "eda/artifacts/instant__lgbm/model_last.txt"))
feat = pl.read_parquet(ROOT / "eda/features.parquet")
tokens = pl.read_parquet(ROOT / "tokens.parquet").select(
    "token_id", "name", "ticker", "description", "twitter_handle", "deployer_address"
)

sample_id = int(feat["token_id"][-1])
row_feat = feat.filter(pl.col("token_id") == sample_id).head(1)
row_tok = tokens.filter(pl.col("token_id") == sample_id).head(1)

MEME_PAT = re.compile(
    "doge|pepe|trump|ai|bonk|cat|shib|moon|elon|grok|maga|inu|bull|pump", re.I
)
TEMPLATES = ["deployed using", "buy now", "to the moon", "100x", "safe", "gem"]
macro = {
    "sol_close": 145.0, "sol_vol_1h": 0.008, "sol_vol_24h": 0.015,
    "sol_ret_1h": 0.003, "sol_ret_24h": -0.012,
    "btc_close": 94500.0, "btc_vol_1h": 0.007, "btc_ret_1h": 0.002,
}

# In production this dict is maintained incrementally as tokens arrive.
deployer_stats: dict = {}

KEEP = [c for c in model.feature_name() if c in feat.columns]
sub = row_feat.select(KEEP)
if "deployer_wallet_source_cex_name" in KEEP:
    sub = sub.with_columns(
        pl.col("deployer_wallet_source_cex_name").fill_null("__missing__").cast(pl.Categorical)
    )
sample_pd = sub.to_pandas()

N = 20_000
deployer_addr = str(row_tok["deployer_address"][0])
name = str(row_tok["name"][0] or "")
desc = str(row_tok["description"][0] or "")
handle = str(row_tok["twitter_handle"][0] or "")


def bench(fn, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


t_lookup = bench(lambda: (
    deployer_stats.get(deployer_addr, {}).get("deployer_hr_7d", 0.0),
    deployer_stats.get(deployer_addr, {}).get("deployer_hr_24h", 0.0),
))

t_text = bench(lambda: (
    1 if MEME_PAT.search(name) else 0,
    len(name.split()),
    sum(c.isdigit() for c in name),
    1 if "http" in desc else 0,
    sum(1 for p in TEMPLATES if p in desc.lower()),
    len(handle),
    sum(c.isdigit() for c in handle) / max(len(handle), 1),
))

t_macro = bench(lambda: (macro["sol_close"], macro["btc_close"]))

pred_holder = [0.0]
t_lgbm = bench(lambda: pred_holder.__setitem__(0, model.predict(sample_pd)[0]))
pred = pred_holder[0]

t_score = bench(lambda: (float(pred) * 100 >= 39.0))

total_us = t_lookup + t_text + t_macro + t_lgbm + t_score
budget_us = 300_000.0

print(f"{'Component':<40} {'μs':>8}  {'ms':>8}")
print("-" * 62)
print(f"{'1. Deployer rolling stats lookup':<40} {t_lookup:8.2f}  {t_lookup/1000:8.4f}")
print(f"{'2. Text features (regex + string)':<40} {t_text:8.2f}  {t_text/1000:8.4f}")
print(f"{'3. Macro context lookup':<40} {t_macro:8.2f}  {t_macro/1000:8.4f}")
print(f"{'4. LGBM single-row predict':<40} {t_lgbm:8.2f}  {t_lgbm/1000:8.4f}")
print(f"{'5. Score + threshold':<40} {t_score:8.2f}  {t_score/1000:8.4f}")
print("-" * 62)
print(f"{'TOTAL':<40} {total_us:8.2f}  {total_us/1000:8.4f}")
print()
print(f"Budget:    {budget_us/1000:.0f} ms")
print(f"Used:      {total_us/1000:.3f} ms")
print(f"Remaining: {(budget_us - total_us)/1000:.1f} ms  ({(budget_us-total_us)/budget_us*100:.1f}% headroom)")
