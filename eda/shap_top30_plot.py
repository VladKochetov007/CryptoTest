"""SHAP top-30 bar chart with 'others' aggregation and cumulative% line."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "eda" / "artifacts" / "meta__lgbm"
shap_path = ART / "shap_meta.json"
out_path = ROOT / "eda" / "plots" / "shap_top30_with_others.png"

records = json.loads(shap_path.read_text())
features = [r["feature"] for r in records]
values = np.array([r["mean_abs_shap"] for r in records])
oof_auc = json.loads((ART / "summary.json").read_text()).get("oof_auc", 0)

TOP_N = 30
top_features = features[:TOP_N]
top_values = values[:TOP_N]
others_sum = values[TOP_N:].sum()
others_count = len(values) - TOP_N

all_labels = top_features + [f"others ({others_count} features)"]
all_values = np.append(top_values, others_sum)

total = all_values.sum()
cum_pct = np.cumsum(all_values) / total * 100

fig, ax1 = plt.subplots(figsize=(10, 11))

colors = ["#4878cf"] * TOP_N + ["#888888"]
y_pos = np.arange(len(all_labels))
bars = ax1.barh(y_pos, all_values, color=colors, height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(all_labels, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlabel("Mean |SHAP|", fontsize=10)
ax1.set_title(f"meta__lgbm — top-30 SHAP features (AUC {oof_auc:.4f})\nBlue = top-30, grey = tail aggregate", fontsize=11)

for bar, val in zip(bars, all_values):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=7.5)

ax2 = ax1.twiny()
ax2.plot(cum_pct, y_pos, color="crimson", linewidth=1.5, marker="o",
         markersize=3, label="Cumulative %")
ax2.set_xlabel("Cumulative % of total |SHAP|", fontsize=10, color="crimson")
ax2.tick_params(axis="x", colors="crimson")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.set_xlim(0, 105)

for x_val, y_val in [(cum_pct[0], y_pos[0]), (cum_pct[9], y_pos[9]),
                      (cum_pct[19], y_pos[19]), (cum_pct[29], y_pos[29])]:
    ax2.annotate(f"{x_val:.0f}%", (x_val, y_val), xytext=(x_val + 1.5, y_val),
                 fontsize=7, color="crimson", va="center")

plt.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
