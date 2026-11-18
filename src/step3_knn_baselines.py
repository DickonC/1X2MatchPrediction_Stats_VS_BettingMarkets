"""
Step 3 (Faris) - K-nearest neighbours on the 3 feature sets + the two baselines (Always-Home, Bookmaker).

Protocol (steps/step3_modelling.md, same for everyone):
  1. fit every hyperparameter candidate on TRAIN (2005/06-2022/23)
  2. pick the candidate with the lowest log loss on VAL (2023/24)
  3. evaluate that same model ONCE on TEST (2024/25 + 2025/26) - no refit on train+val
The baselines are not trained; they are scored on the same test matches with the same metrics.

Baselines
  AlwaysHome : predicts H for every match. Probabilities (for log loss / Brier / AUC) = the TRAIN class
               frequencies (H .460 / D .243 / A .297) for every match -> argmax is H, ROC-AUC = 0.5 by construction.
  Bookmaker  : Bet365 probabilities with the margin removed (ProbHome / ProbDraw / ProbAway); prediction = argmax.

Inputs   data_processed/epl_step2.csv, src/feature_sets.py
Outputs  results/test_metrics.csv                 adds/replaces only rows with owner == "Faris" (Dickon's rows kept)
         results/preds_KNN_<feature_set>.csv      per-match test probabilities (same columns as Dickon's preds files).
                                                  No preds files for the baselines: step3_betting.py reads every
                                                  preds_*.csv as a model and has its own bookmaker / always-home benchmarks.
         results/league_tables_knn.csv            actual vs expected points + rank per team per test season (KNN, Bookmaker)
         logs/step3_knn_tuning.csv                every KNN candidate's val scores
         logs/step3_knn_baselines_run_log.md      chosen settings, val + test metrics, confusion matrices
         figures/step3/knn_k_tuning.png           val log loss vs k per feature set (justifies the chosen k)

Usage:  python src/step3_knn_baselines.py        (~30 s)
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score  # log_loss: sanity check only
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feature_sets import BOOKMAKER_PROBS, CLASSES, FEATURE_SETS, TARGET  # noqa: E402

OWNER = "Faris"
SEED = 42  # KNN itself is deterministic; kept for consistency with the other scripts

# ----------------------------------------------------------------------------- hyperparameter grid
K_VALUES = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500)
KNN_GRID = [{"n_neighbors": k, "weights": w} for w, k in product(("uniform", "distance"), K_VALUES)]


def make_knn(params):
    # distance-based -> features MUST be standardised (Elo is in the 1000s, Form5Diff in single digits);
    # the scaler is fitted on the training rows only, inside the pipeline
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))


# ----------------------------------------------------------------------------- metrics
# proba_hda(), metrics(), season_points() and season_label() are copied unchanged from src/step3_lr_rf.py
# (as agreed in steps/step3_modelling.md) so every model is scored with identical code.
def proba_hda(model, X):
    """predict_proba with columns reordered to CLASSES = [H, D, A] (sklearn sorts them A, D, H)."""
    P = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return P[:, order]


def metrics(y, P):
    y = np.asarray(y)
    pred = np.array(CLASSES)[P.argmax(axis=1)]
    onehot = (y[:, None] == np.array(CLASSES)[None, :]).astype(float)
    # NB: sklearn's log_loss / roc_auc_score assume probability columns in SORTED label order (A, D, H),
    # so with our H, D, A columns we compute them explicitly instead.
    p_true = np.clip(P[onehot == 1], np.finfo(float).eps, 1)  # same clipping as sklearn (a 0% prob for the actual result)
    srt = np.argsort(CLASSES)  # cross-check against sklearn with columns in its expected (sorted) order
    assert np.isclose(-np.mean(np.log(p_true)), log_loss(y, P[:, srt], labels=sorted(CLASSES)))
    assert np.isclose(np.mean([roc_auc_score(onehot[:, k], P[:, k]) for k in range(len(CLASSES))]),
                      roc_auc_score(y, P[:, srt], multi_class="ovr", average="macro", labels=sorted(CLASSES)))
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro"),
        "log_loss": float(-np.mean(np.log(p_true))),                     # natural log
        "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),     # sum over classes, range 0-2
        "roc_auc": float(np.mean([roc_auc_score(onehot[:, k], P[:, k])  # one-vs-rest, macro
                                  for k in range(len(CLASSES))])),
    }


def season_points(matches, P):
    """Points per (Season, Team). With P = [pH, pD, pA] these are EXPECTED points (3*P(win) + 1*P(draw));
    with P = one-hot actual results they are the real league points."""
    home = pd.DataFrame({"Season": matches["Season"], "Team": matches["HomeTeam"], "pts": 3 * P[:, 0] + P[:, 1]})
    away = pd.DataFrame({"Season": matches["Season"], "Team": matches["AwayTeam"], "pts": 3 * P[:, 2] + P[:, 1]})
    return pd.concat([home, away]).groupby(["Season", "Team"])["pts"].sum()


def season_label(start_year):
    return f"{start_year}_{(start_year + 1) % 100:02d}"  # 2024 -> "2024_25"


# ----------------------------------------------------------------------------- data
df = pd.read_csv(ROOT / "data_processed" / "epl_step2.csv")
splits = {s: df[df["Split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
all_features = sorted({f for fs in FEATURE_SETS.values() for f in fs} | set(BOOKMAKER_PROBS))
for s, d in splits.items():
    missing = d[all_features].isna().sum()
    assert missing.sum() == 0, f"missing values in {s}: {missing[missing > 0].to_dict()}"
# strictly chronological: nothing in val/test happens before the end of train/val
dates = {s: pd.to_datetime(d["MatchDate"]) for s, d in splits.items()}
assert dates["train"].max() < dates["val"].min() and dates["val"].max() < dates["test"].min()

for sub in ("results", "logs", "figures/step3"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


log("# Step 3 - KNN + baselines (Faris)\n")
log("Rows: " + ", ".join(f"{s} {len(d)}" for s, d in splits.items()))
log(f"Classes order: {CLASSES}. Selection metric: val log loss. Test evaluated once, no refit.\n")

test_onehot = (splits["test"][TARGET].values[:, None] == np.array(CLASSES)[None, :]).astype(float)
league_tables = {"actual_pts": season_points(splits["test"], test_onehot)}
result_rows, val_reference = [], {}


def record(model_name, fs_name, P_val, P_test, params, champions=True):
    """Score val + test, log them, add a row for results/test_metrics.csv."""
    val = metrics(splits["val"][TARGET], P_val)
    test = metrics(splits["test"][TARGET], P_test)
    champs = {}
    if champions:
        xpts = season_points(splits["test"], P_test)
        league_tables[f"xPts_{model_name}_{fs_name}"] = xpts
        champs = {f"champion_{season_label(s)}": g.idxmax()[1] for s, g in xpts.groupby(level="Season")}
    result_rows.append({"model": model_name, "feature_set": fs_name, "owner": OWNER,
                        **{k: round(v, 4) for k, v in test.items()},
                        "best_params": json.dumps(params) if params else "", **champs})
    yte = splits["test"][TARGET]
    cm = pd.DataFrame(confusion_matrix(yte, np.array(CLASSES)[P_test.argmax(1)], labels=CLASSES),
                      index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    log("val : " + "  ".join(f"{k} {v:.4f}" for k, v in val.items()))
    log("test: " + "  ".join(f"{k} {v:.4f}" for k, v in test.items()))
    log("test confusion matrix:\n```\n" + cm.to_string() + "\n```\n")
    return val


# ----------------------------------------------------------------------------- baseline 1: Always-Home
train_freq = splits["train"][TARGET].value_counts(normalize=True).reindex(CLASSES).to_numpy()
assert train_freq.argmax() == CLASSES.index("H")  # so the predicted class is always H
log("## AlwaysHome (baseline, no training)")
log("probabilities = train class frequencies for every match: "
    + ", ".join(f"{c} {p:.3f}" for c, p in zip(CLASSES, train_freq)))
P_val_home = np.tile(train_freq, (len(splits["val"]), 1))
P_test_home = np.tile(train_freq, (len(splits["test"]), 1))
val_reference["AlwaysHome"] = record("AlwaysHome", "none", P_val_home, P_test_home, None, champions=False)
home_share = (splits["test"][TARGET] == "H").mean()
assert np.isclose(result_rows[-1]["accuracy"], round(home_share, 4)), "Always-Home accuracy must equal the test home-win share"

# ----------------------------------------------------------------------------- baseline 2: Bookmaker (Bet365)
log("## Bookmaker (baseline, no training)")
log(f"probabilities = {BOOKMAKER_PROBS} (Bet365, margin removed)")
P_val_book = splits["val"][BOOKMAKER_PROBS].to_numpy()
P_test_book = splits["test"][BOOKMAKER_PROBS].to_numpy()
for P in (P_val_book, P_test_book):
    assert np.allclose(P.sum(axis=1), 1), "bookmaker probabilities must sum to 1"
val_reference["Bookmaker"] = record("Bookmaker", "bet365", P_val_book, P_test_book, None)

# ----------------------------------------------------------------------------- KNN: tune on val, evaluate on test
tuning_rows = []
for fs_name, feats in FEATURE_SETS.items():
    Xtr, ytr = splits["train"][feats], splits["train"][TARGET]
    Xva, yva = splits["val"][feats], splits["val"][TARGET]
    Xte = splits["test"][feats]

    t0 = time.time()
    best = None
    for params in KNN_GRID:
        m = make_knn(params).fit(Xtr, ytr)
        P_val = proba_hda(m, Xva)
        val = metrics(yva, P_val)
        tuning_rows.append({"model": "KNN", "feature_set": fs_name, "params": json.dumps(params),
                            **params, **{f"val_{k}": v for k, v in val.items()}})
        if best is None or val["log_loss"] < best[1]["log_loss"]:
            best = (params, val, m, P_val)

    params, val, m, P_val = best
    P_test = proba_hda(m, Xte)  # the ONLY time test features are used, with the model trained on train only
    log(f"## KNN-{fs_name}  ({len(KNN_GRID)} candidates, {time.time() - t0:.0f}s)")
    log(f"best params: {params}")
    record("KNN", fs_name, P_val, P_test, params)
    pd.DataFrame({"match_id": splits["test"]["match_id"], "Season": splits["test"]["Season"],
                  "HomeTeam": splits["test"]["HomeTeam"], "AwayTeam": splits["test"]["AwayTeam"],
                  "FTResult": splits["test"][TARGET], "pH": P_test[:, 0], "pD": P_test[:, 1], "pA": P_test[:, 2]}
                 ).to_csv(ROOT / "results" / f"preds_KNN_{fs_name}.csv", index=False)

tuning = pd.DataFrame(tuning_rows)
tuning.to_csv(ROOT / "logs" / "step3_knn_tuning.csv", index=False)

# ----------------------------------------------------------------------------- save
out = ROOT / "results" / "test_metrics.csv"
new = pd.DataFrame(result_rows)
if out.exists():  # keep other people's rows (e.g. Dickon's LR/RF), replace only ours
    old = pd.read_csv(out)
    new = pd.concat([old[old["owner"] != OWNER], new], ignore_index=True)
new.to_csv(out, index=False)

tables = pd.DataFrame(league_tables).round(1).reset_index()
for col in [c for c in tables.columns if c.endswith("pts") or c.startswith("xPts")]:
    rank_col = "actual_rank" if col == "actual_pts" else col.replace("xPts", "rank")
    tables[rank_col] = tables.groupby("Season")[col].rank(ascending=False, method="min").astype(int)
tables = tables.sort_values(["Season", "actual_rank"])
tables.to_csv(ROOT / "results" / "league_tables_knn.csv", index=False)

# ----------------------------------------------------------------------------- figure: val log loss vs k
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FS_STYLE = {  # dataviz reference palette, categorical slots 1-3 (validated); marker shape = secondary encoding
    "non_market": ("#2a78d6", "o", "Non-market"),
    "market": ("#eb6834", "^", "Market"),
    "combined": ("#1baf7a", "s", "Combined"),
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.grid(axis="y", color=GRID, linewidth=0.6)
chosen = {r["feature_set"]: json.loads(r["best_params"]) for r in result_rows if r["model"] == "KNN"}
ends = []
for fs_name, (color, marker, label) in FS_STYLE.items():
    w = chosen[fs_name]["weights"]
    curve = tuning[(tuning.feature_set == fs_name) & (tuning.weights == w)].sort_values("n_neighbors")
    k_best = chosen[fs_name]["n_neighbors"]
    ax.plot(curve.n_neighbors, curve.val_log_loss, color=color, marker=marker, markersize=5, linewidth=2,
            label=f"{label}: chosen k = {k_best} ({w})")
    y_best = curve.loc[curve.n_neighbors == k_best, "val_log_loss"].iloc[0]
    ax.plot([k_best], [y_best], marker=marker, markersize=11, color=color, markeredgecolor=INK, zorder=5)
    ends.append([curve.val_log_loss.iloc[-1], label])
for name, style in (("Bookmaker", "--"), ("AlwaysHome", ":")):
    ax.axhline(val_reference[name]["log_loss"], color=INK2, linestyle=style, linewidth=1.2)
    ends.append([val_reference[name]["log_loss"], "Bookmaker (Bet365)" if name == "Bookmaker" else "Always-Home (class prior)"])
ref = [v["log_loss"] for v in val_reference.values()]
y_lo = min(tuning.val_log_loss.min(), *ref) - 0.01
y_hi = max(ref) + 0.04
ax.set_ylim(y_lo, y_hi)
# direct labels at the right edge, nudged apart so they never overlap
ends.sort(key=lambda e: e[0])
min_gap = (y_hi - y_lo) * 0.055
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
for y, text in ends:
    ax.text(560, y, text, va="center", ha="left", fontsize=8, color=INK)
ax.set_xscale("log")
TICKS = (5, 10, 20, 50, 100, 200, 500)
ax.set_xticks(TICKS)
ax.set_xticklabels([str(k) for k in TICKS])
ax.minorticks_off()
ax.set_xlim(4, 540)
ax.set_xlabel("k (number of neighbours, log scale)")
ax.set_ylabel("Validation log loss (2023/24)  - lower is better")
ax.set_title("KNN: choosing k on the validation season", loc="left", fontsize=10, color=INK, pad=20)
ax.text(0, 1.015, "Large marker = chosen k. Lines leaving the top edge: small k gives 0% probabilities, "
        "which log loss punishes heavily.", transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(0, -0.15))
fig.subplots_adjust(right=0.74, bottom=0.3, top=0.86)
fig.savefig(ROOT / "figures" / "step3" / "knn_k_tuning.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- summary
log("## Predicted champions (most expected points over the season)")
actual_champs = {f"champion_{season_label(s)}": g.loc[g["actual_rank"] == 1, "Team"].iloc[0]
                 for s, g in tables.groupby("Season")}
log("Actual: " + ", ".join(f"{k[9:]} {v}" for k, v in actual_champs.items()))
log("```\n" + pd.DataFrame(result_rows)[["model", "feature_set", *actual_champs]].fillna("-").to_string(index=False)
    + "\n```\n")
log("## Test summary - all owners (results/test_metrics.csv)\n```\n"
    + new.drop(columns="best_params").fillna("-").to_string(index=False) + "\n```")
(ROOT / "logs" / "step3_knn_baselines_run_log.md").write_text("\n".join(log_lines), encoding="utf-8")
