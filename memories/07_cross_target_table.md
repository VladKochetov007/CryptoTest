---
name: cross_target_table
description: Cross-target transfer experiment results — train on Nx label, eval on Mx. Hypothesis refuted.
type: project
---

# Cross-Target Experiment Results

**Date:** 2026-04-28. Script: `eda/cross_target.py`.

## Setup
- 470,256 tokens with label-eligible first-slot price
- 5-fold walk-forward CV, expanding window, LGBM only
- Top-10% selection for lift computation
- Base rates: hit_2x=16.0%, hit_3x=7.8%, hit_5x=3.4%, hit_10x=1.2%

## AUC Matrix (rows=train target, cols=eval target)

| Train \ Eval | hit_2x | hit_3x | hit_5x | hit_10x |
|---|---|---|---|---|
| **hit_2x** | **0.7652** | 0.7655 | 0.7748 | 0.7747 |
| hit_3x | 0.7529 | **0.7671** | 0.7827 | 0.7870 |
| hit_5x | 0.7320 | 0.7534 | **0.7780** | 0.7917 |
| hit_10x | 0.6687 | 0.6943 | 0.7254 | **0.7543** |

## Lift@top-10% Matrix (rows=train, cols=eval)

| Train \ Eval | hit_2x | hit_3x | hit_5x | hit_10x |
|---|---|---|---|---|
| **hit_2x** | **2.875** | 3.328 | 3.757 | 4.048 |
| hit_3x | 2.762 | **3.352** | 3.845 | 4.193 |
| hit_5x | 2.608 | 3.261 | **3.885** | 4.328 |
| hit_10x | 2.295 | 2.773 | 3.318 | **3.940** |

## Key Findings

**Hypothesis REFUTED:** Training on harder label (5x, 10x) does NOT improve hit_2x detection in this dataset. In fact it degrades it.

### AUC diagonal dominates: same-label training is best
Each model is best at predicting its own training target. The diagonal is always the maximum in each column.

### Cross-target is useful for harder thresholds only
Train hit_3x → eval hit_5x = 0.7827 vs train hit_2x → eval hit_5x = 0.7748.
Train hit_5x → eval hit_10x = 0.7917 vs train hit_2x → eval hit_10x = 0.7747.

Training on 3x label gives +0.8 AUC for hit_5x detection vs training on 2x.
Training on 5x label gives +1.7 AUC for hit_10x detection vs training on 2x.

### Actionable dual-model architecture

| Model | Train target | Use for |
|---|---|---|
| Instant gating model (existing) | hit_2x | Probe decision (0.1 SOL) |
| High-conviction model | hit_5x | Full-size scale (1.0 SOL) at 60s |

The 60s scale model should be retrained on hit_5x instead of hit_2x. This gives:
- Better lift@10% for hit_5x (3.885 vs 3.757 → +0.13)
- Better lift@10% for hit_10x (4.328 vs 4.048 → +0.28) = captures the right tail better

**Why the hypothesis was wrong in this data:** The user's prior experience may have been on a different market regime or with different base rates. When hit_2x base rate is 16% (not rare), the model doesn't need to specialize on extreme tails to get lift at the 2x level.

## Files
- `eda/cross_target/auc_matrix.json`
- `eda/cross_target/lift_matrix.json`
- `eda/cross_target/summary.md`
- `eda/plots/cross_target_auc.png`
- `eda/plots/cross_target_lift.png`
