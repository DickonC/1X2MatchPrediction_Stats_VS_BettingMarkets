"""
Step 2 - Variable refinement: redundancy (Spearman) -> relevance (feature vs class) -> Boruta.

All selection decisions use the TRAIN split only (2005/06-2022/23).

Inputs   data_processed/epl_step1.csv
Outputs  data_processed/epl_step2.csv          step-1 data + train-fitted imputations
         data_processed/features_step2.json    final feature lists (A = non-market, AM = A + market)
         logs/step2_imputations.csv            every imputation (column, split, cells, fill value)
         logs/step2_feature_decisions.csv      every candidate feature -> kept/dropped + stage + reason
         logs/step2_run_log.md                 diagnostics from the run
         figures/step2/*.png                   correlation matrix, boxplots, relevance ranking, Boruta

Usage:  python src/step2_refine.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
LOGS = ROOT / "logs"
FIGS = ROOT / "figures" / "step2"
FIGS.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "corr_threshold": 0.85,   # |Spearman rho| at/above which two features count as redundant
    "alpha": 0.05,            # Kruskal-Wallis significance (Bonferroni-corrected)
    "mi_null_perms": 20,      # shuffled-target runs to estimate the "no information" MI level
    "mi_seeds": 5,            # MI is a kNN estimate -> average over seeds for stability
    "boruta_max_iter": 200,
    "boruta_seeds": (42, 7, 2026),  # final status = majority vote across seeds (stability)
    "vif_threshold": 10,      # 2.4: drop features until every VIF is below this
    "seed": 42,
}

# ----------------------------------------------------------------------------- style (dataviz reference palette)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {"H": "#2a78d6", "D": "#eb6834", "A": "#1baf7a"}  # categorical slots 1-3 (validated all-pairs)
CLASS_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
DECISION_COLORS = {"Confirmed": "#2a78d6", "Tentative": "#eda100", "Rejected": "#b5b4ae"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#b8312f", "#e34948", "#f0efec", "#3987e5", "#184f95"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})

log_lines, decisions = [], []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def decide(feature, group, stage, kept, reason):
    decisions.append({"feature": feature, "group": group, "stage": stage, "kept": kept, "reason": reason})


# ----------------------------------------------------------------------------- feature groups
A_FEATURES = (
    ["HomeElo", "AwayElo", "EloDiff", "EloChange1Home", "EloChange2Home", "EloChange1Away", "EloChange2Away",
     "Form3Home", "Form5Home", "Form3Away", "Form5Away", "Form3Diff", "Form5Diff"]
    + [f"{s}_Avg5_{st}{d}" for s in ("Home", "Away")
       for st in ("Goals", "Shots", "Target", "Corners", "Fouls", "Yellow", "Red") for d in ("For", "Against")]
    + ["Home_H2H_n", "Home_H2H_PPG", "Home_H2H_GDPG"]
)
M_FEATURES = ["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
              "Over25", "Under25", "MaxOver25", "MaxUnder25", "HandiSize", "HandiHome", "HandiAway",
              "BookmakerMargin", "ProbHome", "ProbDraw", "ProbAway"]
GROUP = {**{f: "A (non-market)" for f in A_FEATURES}, **{f: "M (market)" for f in M_FEATURES}}

df = pd.read_csv(DATA / "epl_step1.csv", index_col="match_id", parse_dates=["MatchDate"])
train = df["Split"] == "train"
log(f"Loaded {df.shape}; train rows = {train.sum()} (seasons {df.loc[train, 'Season'].min()}-{df.loc[train, 'Season'].max()})")
log(f"Candidates: {len(A_FEATURES)} non-market (A) + {len(M_FEATURES)} market (M) = {len(GROUP)}")

# ----------------------------------------------------------------------------- 2.0 imputation (fit on train only)
imputations = []


def impute(col, value, reason):
    for split, idx in df.groupby("Split").groups.items():
        n = int(df.loc[idx, col].isna().sum())
        if n:
            imputations.append({"column": col, "split": split, "cells": n, "fill_value": value, "reason": reason})
    df[col] = df[col].fillna(value)


log("\n## 2.0 Imputation")
impute("Home_H2H_GDPG", 0.0, "teams never met before: no H2H evidence -> neutral goal difference 0 (Home_H2H_n = 0 flags it)")
impute("Home_H2H_PPG", round(df.loc[train, "Home_H2H_PPG"].mean(), 4),
       "teams never met before: fill with train mean PPG (Home_H2H_n = 0 flags it)")
for col in ("HandiSize", "HandiHome", "HandiAway"):
    impute(col, float(df.loc[train, col].median()), "handicap odds missing for a handful of matches: train median")
imp = pd.DataFrame(imputations)
log(imp.to_string(index=False))
assert not df[list(GROUP)].isna().any().any(), "unexpected missing values remain"

X = df.loc[train, list(GROUP)]
y = df.loc[train, "FTResult"]

# ----------------------------------------------------------------------------- relevance scores (used by 2.1 tie-break and 2.2)
log("\n## Relevance scores (train)")
mi = np.mean([mutual_info_classif(X, y, random_state=s) for s in range(CONFIG["mi_seeds"])], axis=0)
mi = pd.Series(mi, index=X.columns)
rng = np.random.default_rng(CONFIG["seed"])
mi_null = np.array([mutual_info_classif(X, rng.permutation(y.values), random_state=i)
                    for i in range(CONFIG["mi_null_perms"])])
mi_null95 = pd.Series(np.percentile(mi_null, 95, axis=0), index=X.columns)

k, n = y.nunique(), len(y)
kw = {}
for f in X.columns:
    H, p = kruskal(*[X.loc[y == c, f] for c in ("H", "D", "A")])
    kw[f] = {"KW_H": H, "p_bonf": min(1.0, p * len(X.columns)), "eps2": max(0.0, (H - k + 1) / (n - k))}
rel = pd.DataFrame(kw).T.assign(MI=mi, MI_null95=mi_null95)
rel["group"] = rel.index.map(GROUP)

# ----------------------------------------------------------------------------- 2.1 redundancy (Spearman)
log(f"\n## 2.1 Redundancy: Spearman |rho| >= {CONFIG['corr_threshold']}, clustered within each group")
rho = X.corr(method="spearman")


def redundancy_clusters(features):
    """Connected components of the graph 'features with |rho| >= threshold'."""
    adj = (rho.loc[features, features].abs() >= CONFIG["corr_threshold"])
    seen, clusters = set(), []
    for f in features:
        if f in seen:
            continue
        stack, comp = [f], []
        while stack:
            g = stack.pop()
            if g in seen:
                continue
            seen.add(g)
            comp.append(g)
            stack.extend(adj.columns[adj.loc[g]].difference(seen))
        clusters.append(comp)
    return clusters


after_corr = []
for group_features in (A_FEATURES, M_FEATURES):
    for comp in redundancy_clusters(group_features):
        if len(comp) == 1:
            after_corr.append(comp[0])
            continue
        keep = rel.loc[comp, "MI"].idxmax()  # keep the cluster member that carries the most information about FTResult
        after_corr.append(keep)
        log(f"- cluster {comp} -> keep {keep}")
        for f in comp:
            if f != keep:
                r = rho.loc[f, keep]
                decide(f, GROUP[f], "2.1 correlation", False,
                       f"redundant with {keep} (rho={r:+.2f}); {keep} has higher MI with result")
log(f"{len(GROUP)} -> {len(after_corr)} features after redundancy step")

cross = (rho.loc[A_FEATURES, M_FEATURES].abs().stack().sort_values(ascending=False))
log("\nStrongest A-vs-M correlations (reported, not used to drop - we need A on its own):")
log(cross.head(6).round(3).to_string())

# figure: full correlation matrix, grouped A | M, hierarchically ordered within each group
def order_within(features):
    d = 1 - rho.loc[features, features].abs().values
    np.fill_diagonal(d, 0)
    return [features[i] for i in leaves_list(linkage(squareform(d, checks=False), "average"))]


order = order_within(A_FEATURES) + order_within(M_FEATURES)
R = rho.loc[order, order]
fig, ax = plt.subplots(figsize=(15, 13.5))
im = ax.imshow(R.values, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
ax.set_yticks(range(len(order)), order, fontsize=7)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    if lbl.get_text() not in after_corr:
        lbl.set_color("#a3a29b")  # greyed label = dropped as redundant
for i in range(len(order)):
    for j in range(len(order)):
        if i != j and abs(R.values[i, j]) >= CONFIG["corr_threshold"]:
            ax.text(j, i, f"{R.values[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.5,
                    color="white" if abs(R.values[i, j]) > 0.9 else INK)
nA = len(A_FEATURES)
ax.axhline(nA - 0.5, color=INK, lw=1)
ax.axvline(nA - 0.5, color=INK, lw=1)
ax.text(nA / 2, -1.2, "A: non-market", ha="center", fontsize=10, fontweight="bold")
ax.text(nA + len(M_FEATURES) / 2, -1.2, "M: market (odds)", ha="center", fontsize=10, fontweight="bold")
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cb.set_label("Spearman ρ", color=INK2)
cb.outline.set_visible(False)
ax.set_title(f"Spearman correlation of candidate pre-match features (train 2005/06–2022/23, n={n})\n"
             f"Cells with |ρ| ≥ {CONFIG['corr_threshold']} are annotated · grey labels = dropped as redundant",
             loc="left", fontsize=11, pad=28)
fig.tight_layout()
fig.savefig(FIGS / "01_correlation_matrix.png", dpi=160)
plt.close(fig)

# ----------------------------------------------------------------------------- 2.2 relevance: feature vs class
log(f"\n## 2.2 Feature vs class (Kruskal-Wallis, Bonferroni alpha={CONFIG['alpha']}; MI vs shuffled-target 95th pct)")
after_rel = []
for f in after_corr:
    r = rel.loc[f]
    sig = r.p_bonf < CONFIG["alpha"]
    informative = r.MI > r.MI_null95
    if sig or informative:
        after_rel.append(f)
    else:
        decide(f, GROUP[f], "2.2 feature vs class", False,
               f"no detectable class difference: KW p_bonf={r.p_bonf:.2f}, eps2={r.eps2:.4f}, "
               f"MI={r.MI:.4f} <= null {r.MI_null95:.4f}")
log(rel.loc[after_corr, ["group", "eps2", "p_bonf", "MI", "MI_null95"]]
    .sort_values("eps2", ascending=False).round(4).to_string())
log(f"{len(after_corr)} -> {len(after_rel)} features after relevance step")

# figure: boxplots by class, ordered by effect size
plot_feats = rel.loc[after_corr].sort_values("eps2", ascending=False).index.tolist()
ncol = 6
nrow = int(np.ceil(len(plot_feats) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.3))
for ax, f in zip(axes.flat, plot_feats):
    data = [X.loc[y == c, f] for c in ("H", "D", "A")]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0))
    for patch, c in zip(bp["boxes"], ("H", "D", "A")):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.85)
    for i, d in enumerate(data, start=1):
        ax.plot(i, d.mean(), marker="D", ms=3.5, color=SURFACE, mec=INK, mew=0.7)
    dropped = f not in after_rel
    ax.set_title(f, fontsize=8.5, color="#a3a29b" if dropped else INK, loc="left")
    ax.text(0.98, 0.97, f"ε²={rel.at[f, 'eps2']:.3f}" + (" ✗" if dropped else ""), transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xticks([1, 2, 3], ["H", "D", "A"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
for ax in axes.flat[len(plot_feats):]:
    ax.axis("off")
handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in ("H", "D", "A")]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("Feature distribution by match result (train) – boxes = IQR, line = median, ◆ = mean, whiskers 1.5×IQR\n"
             "Sorted by Kruskal–Wallis effect size ε² · grey title + ✗ = no detectable class difference (dropped)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(FIGS / "02_boxplots_by_result.png", dpi=150)
plt.close(fig)

# figure: relevance ranking (eps2 and MI side by side, one axis each)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.24 * len(plot_feats) + 1.6), sharey=True)
ypos = np.arange(len(plot_feats))[::-1]
colors = ["#2a78d6" if GROUP[f].startswith("A") else "#eb6834" for f in plot_feats]
ax1.barh(ypos, rel.loc[plot_feats, "eps2"], color=colors, height=0.7)
ax1.set_yticks(ypos, plot_feats, fontsize=7.5)
ax1.set_xlabel("Kruskal–Wallis effect size ε²")
ax2.barh(ypos, rel.loc[plot_feats, "MI"], color=colors, height=0.7)
ax2.scatter(rel.loc[plot_feats, "MI_null95"], ypos, marker="|", s=60, color=INK, zorder=3,
            label="95th pct of MI with shuffled result (chance level)")
ax2.set_xlabel("Mutual information with result (nats)")
for ax in (ax1, ax2):
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
for lbl in ax1.get_yticklabels():
    if lbl.get_text() not in after_rel:
        lbl.set_color("#a3a29b")
ax2.legend(handles=[Patch(color="#2a78d6", label="A: non-market"), Patch(color="#eb6834", label="M: market"),
                    ax2.collections[0]], frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Relevance of each feature to the match result (train) – grey labels dropped at step 2.2",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "03_relevance_ranking.png", dpi=160)
plt.close(fig)

# ----------------------------------------------------------------------------- 2.3 Boruta
class RecordingBoruta(BorutaPy):
    """BorutaPy that also records the max shadow importance of each iteration (for plotting)."""

    def _add_shadows_get_imps(self, X, y, dec_reg):
        imp_real, imp_sha = super()._add_shadows_get_imps(X, y, dec_reg)
        self.shadow_max_history_.append(np.percentile(imp_sha, self.perc))
        return imp_real, imp_sha


def run_boruta(features, label):
    """Run Boruta once per seed; final status = majority vote. History/shadows returned for the first seed."""
    runs = {}
    for seed in CONFIG["boruta_seeds"]:
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=7, random_state=seed)
        b = RecordingBoruta(rf, n_estimators="auto", max_iter=CONFIG["boruta_max_iter"], random_state=seed, verbose=0)
        b.shadow_max_history_ = []
        b.fit(X[features].values, y.values)
        runs[seed] = pd.Series(np.where(b.support_, "Confirmed", np.where(b.support_weak_, "Tentative", "Rejected")),
                               index=features)
        if seed == CONFIG["boruta_seeds"][0]:
            hist = pd.DataFrame(b.importance_history_[1:], columns=features)
            shadow = np.array(b.shadow_max_history_)
    runs = pd.DataFrame(runs)
    n_conf = (runs == "Confirmed").sum(axis=1)
    n_rej = (runs == "Rejected").sum(axis=1)
    half = len(CONFIG["boruta_seeds"]) / 2
    status = pd.Series(np.where(n_conf > half, "Confirmed", np.where(n_rej > half, "Rejected", "Tentative")),
                       index=features)
    unstable = runs.index[runs.nunique(axis=1) > 1].tolist()
    log(f"\nBoruta on {label} ({len(features)} features, seeds {CONFIG['boruta_seeds']}): "
        + ", ".join(f"{s} {int((status == s).sum())}" for s in ("Confirmed", "Tentative", "Rejected"))
        + f"; not unanimous across seeds: {unstable or 'none'}")
    return status, hist, shadow, n_conf


def plot_boruta(status, hist, shadow, title, path):
    med = hist.median().sort_values()
    feats = med.index.tolist()
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(feats) + 1.8))
    data = [hist[f].dropna().values for f in feats]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=INK2, lw=0.7),
                    capprops=dict(color=INK2, lw=0.7), boxprops=dict(lw=0))
    for patch, f in zip(bp["boxes"], feats):
        patch.set_facecolor(DECISION_COLORS[status[f]])
    lo, hi = np.percentile(shadow, [25, 75])
    ax.axvspan(lo, hi, color="#b5b4ae", alpha=0.25, lw=0)
    ax.axvline(np.median(shadow), color=INK, ls="--", lw=1)
    ax.set_yticks(range(1, len(feats) + 1), feats, fontsize=7.5)
    ax.set_xlabel("Random-forest importance per Boruta iteration")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(color=DECISION_COLORS[s], label=f"{s} ({int((status == s).sum())})")
               for s in DECISION_COLORS]
    handles.append(plt.Line2D([], [], color=INK, ls="--", label="best shadow feature (median, band = IQR)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


log("\n## 2.3 Boruta (RandomForest, class_weight=balanced, max_depth=7)")
A_in = [f for f in after_rel if GROUP[f].startswith("A")]
status_A, hist_A, sh_A, nconf_A = run_boruta(A_in, "A (non-market)")
status_AM, hist_AM, sh_AM, nconf_AM = run_boruta(after_rel, "A+M (non-market + market)")
nseed = len(CONFIG["boruta_seeds"])
log("\n" + pd.DataFrame({"A run": status_A, "A conf/seeds": nconf_A.astype(str) + f"/{nseed}",
                         "A+M run": status_AM, "A+M conf/seeds": nconf_AM.astype(str) + f"/{nseed}"})
    .fillna("-").to_string())
plot_boruta(status_A, hist_A, sh_A, "Boruta – non-market features only (A)\n"
            "A feature is confirmed if it beats the best shuffled 'shadow' copy significantly often",
            FIGS / "04_boruta_A.png")
plot_boruta(status_AM, hist_AM, sh_AM, "Boruta – non-market + market features (A+M)\n"
            "A feature is confirmed if it beats the best shuffled 'shadow' copy significantly often",
            FIGS / "05_boruta_AM.png")

for f in after_rel:
    in_A = f in status_A.index
    if (not in_A or status_A[f] != "Confirmed") and status_AM[f] != "Confirmed":
        a_txt = f"A run: {status_A[f]} ({nconf_A[f]}/{nseed} seeds); " if in_A else ""
        decide(f, GROUP[f], "2.3 Boruta", False,
               f"{a_txt}A+M run: {status_AM[f]} ({nconf_AM[f]}/{nseed} seeds)")

# ----------------------------------------------------------------------------- 2.4 multicollinearity (VIF)
# Pairwise correlation misses linear combinations (e.g. EloDiff = HomeElo - AwayElo exactly), which make
# linear models (logistic regression) unidentifiable. Iteratively drop the high-VIF feature that Boruta
# found least important until all VIF < threshold.
def vif(features):
    """VIF_j = 1 / (1 - R^2_j), regressing feature j on the others (inf for exact linear dependence)."""
    Z = X[features].astype(float)
    Z = ((Z - Z.mean()) / Z.std()).values
    out = {}
    for j, f in enumerate(features):
        others = np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        r2 = 1 - resid.var() / Z[:, j].var()
        out[f] = np.inf if 1 - r2 < 1e-9 else 1 / (1 - r2)
    return pd.Series(out)


def vif_prune(features, importance, label):
    feats = list(features)
    log(f"\n{label}: VIF before pruning:\n" + vif(feats).round(1).sort_values(ascending=False).to_string())
    while True:
        v = vif(feats)
        high = v[v >= CONFIG["vif_threshold"]]
        if high.empty:
            break
        drop = importance[high.index].idxmin()
        log(f"- drop {drop} (VIF {v[drop]:.1f}; lowest Boruta importance among VIF>={CONFIG['vif_threshold']}: "
            f"{high.index.tolist()})")
        decide(drop, GROUP[drop], "2.4 VIF", False,
               f"[{label}] multicollinear (VIF={v[drop]:.0f}) with {[f for f in high.index if f != drop]}; "
               f"lowest Boruta importance of those")
        feats.remove(drop)
    log(f"{label}: max VIF after pruning = {vif(feats).max():.1f}")
    return feats


log("\n## 2.4 Multicollinearity (VIF)")
conf_A = [f for f in A_in if status_A[f] == "Confirmed"]
conf_AM = [f for f in after_rel if status_AM[f] == "Confirmed"]
final_A = vif_prune(conf_A, hist_A.median(), "A")
final_AM = vif_prune(conf_AM, hist_AM.median(), "AM")
for f in sorted(set(final_A) | set(final_AM), key=list(GROUP).index):
    decide(f, GROUP[f], "final", True, "kept in " + " and ".join(
        s for s, lst in (("A", final_A), ("AM", final_AM)) if f in lst))

# ----------------------------------------------------------------------------- outputs
features = {
    "A": final_A,
    "AM": final_AM,
    "A_boruta_before_vif": conf_A,    # fine for tree models, which don't suffer from multicollinearity
    "AM_boruta_before_vif": conf_AM,
    "tentative": sorted(set(status_A.index[status_A == "Tentative"]) | set(status_AM.index[status_AM == "Tentative"])),
    "notes": "A = Boruta-confirmed (majority of seeds) in the non-market-only run, then VIF-pruned; "
             "AM = same for the non-market + market run.",
}
(DATA / "features_step2.json").write_text(json.dumps(features, indent=2))
df.to_csv(DATA / "epl_step2.csv")
imp.to_csv(LOGS / "step2_imputations.csv", index=False)
dec = pd.DataFrame(decisions).set_index("feature").join(
    rel[["eps2", "p_bonf", "MI", "MI_null95"]].round(4))
dec.to_csv(LOGS / "step2_feature_decisions.csv")
log(f"\nFINAL A  ({len(final_A)}): {final_A}")
log(f"FINAL AM ({len(final_AM)}): {final_AM}")
log(f"Tentative: {features['tentative']}")
(LOGS / "step2_run_log.md").write_text("# Step 2 run log (auto-generated)\n\n```\n" + "\n".join(log_lines) + "\n```\n")
