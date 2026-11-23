"""
Step 3 - summary figure: test log loss for every model x feature set, against the two baselines.

Answers the research question at a glance:
  - every model sits well left of Always-Home          -> pre-kick-off info predicts results
  - market (orange) is left of non-market (blue) per row -> market information helps
  - combined (green) is no better than market            -> our features add nothing beyond the market
  - nothing crosses the Bookmaker line                   -> no model beats the bookmaker

Inputs   results/test_metrics.csv (Dickon's LR/RF rows + Faris's KNN/baseline rows)
Output   figures/step3/test_log_loss.png
Usage    python src/step3_figures.py
Style matches figures/step3/knn_k_tuning.png (same colours + marker shape per feature set).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results" / "test_metrics.csv")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FS_STYLE = {  # same validated slots as src/step3_knn_baselines.py; marker shape = secondary encoding
    "non_market": ("#2a78d6", "o", "Non-market (9 features)"),
    "market": ("#eb6834", "^", "Market (3 features)"),
    "combined": ("#1baf7a", "s", "Combined (12 features)"),
}
MODELS = [("LR", "Logistic regression"), ("RF", "Random forest"), ("KNN", "k-nearest neighbours")]
DODGE = {"non_market": -0.22, "market": 0.0, "combined": 0.22}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "font.size": 9,
})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)

# baselines: vertical reference lines, labelled at the top
ref = {r.model: r.log_loss for r in m[m.model.isin(["Bookmaker", "AlwaysHome"])].itertuples()}
for name, style, text in (("Bookmaker", "--", "Bookmaker (Bet365)"), ("AlwaysHome", ":", "Always-Home")):
    ax.axvline(ref[name], color=INK2, linestyle=style, linewidth=1.2, zorder=1)
    # y axis is inverted (LR on top), so -0.6 is just above the top row; +1e-9 = round half up (1.0835 -> 1.084)
    ax.text(ref[name], -0.6, f"{text}  {ref[name] + 1e-9:.3f}", ha="center", va="bottom", fontsize=8,
            color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

# one row per model family, three dodged markers (one per feature set)
for row, (model, label) in enumerate(MODELS):
    sub = m[m.model == model].set_index("feature_set")
    xs = sub.log_loss
    ax.plot([xs.min(), xs.max()], [row, row], color=GRID, linewidth=1, zorder=2)  # spread within the family
    for fs, (color, marker, _) in FS_STYLE.items():
        x = sub.loc[fs, "log_loss"]
        ax.plot(x, row + DODGE[fs], marker=marker, markersize=9, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, linestyle="none", zorder=4)
    best = sub.log_loss.idxmin()  # selective label: best feature set per model only
    ax.text(sub.loc[best, "log_loss"] - 0.0025, row + DODGE[best], f"{sub.loc[best, 'log_loss']:.3f}",
            ha="right", va="center", fontsize=8, color=INK)

ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([l for _, l in MODELS], color=INK)
ax.tick_params(axis="y", length=0)
ax.set_ylim(2.45, -0.62)  # LR on top
ax.set_xlim(0.975, 1.095)
ax.set_xlabel("Test log loss (2024/25 + 2025/26, 760 matches)  - lower is better")
ax.set_title("Market information helps every model, but none beats the bookmaker", loc="left",
             fontsize=10, color=INK, pad=24)
for fs, (color, marker, text) in FS_STYLE.items():
    ax.plot([], [], marker=marker, color=color, markersize=7, linestyle="none", label=text)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False, fontsize=8,
          handletextpad=0.3, columnspacing=1.4)
fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.27)
out = ROOT / "figures" / "step3" / "test_log_loss.png"
fig.savefig(out, dpi=200)
print("saved", out)
