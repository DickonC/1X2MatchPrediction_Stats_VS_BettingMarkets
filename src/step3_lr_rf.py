"""
Step 3 (Dickon) - Logistic regression + Random forest on the 3 feature sets (6 models).

Protocol (steps/step3_modelling.md, same for everyone):
  1. fit every hyperparameter candidate on TRAIN (2005/06-2022/23)
  2. pick the candidate with the lowest log loss on VAL (2023/24)
  3. evaluate that same model ONCE on TEST (2024/25 + 2025/26) - no refit on train+val

Inputs   data_processed/epl_step2.csv
Outputs  results/test_metrics.csv                 one row per model x feature set (shared with Faris:
                                                  only rows with owner == "Dickon" are replaced)
         results/preds_<model>_<feature_set>.csv  per-match test probabilities (match_id, Season, teams, FTResult, pH, pD, pA)
         results/league_tables_lr_rf.csv          actual vs expected points + rank per team per test season, per model
                                                  (champion_<season> columns in test_metrics.csv = top of each table)
         logs/step3_lr_rf_tuning.csv              every candidate's val scores
         logs/step3_lr_rf_run_log.md              chosen settings, val + test metrics, confusion matrices

Usage:  python src/step3_lr_rf.py        (~3 min, mostly the RF grid)
        On Dickon's PC: C:\\ProgramData\\anaconda3\\python.exe src\\step3_lr_rf.py
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score  # log_loss: sanity check only
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feature_sets import CLASSES, FEATURE_SETS, TARGET  # noqa: E402

OWNER = "Dickon"
SEED = 42

# ----------------------------------------------------------------------------- hyperparameter grids
LR_GRID = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100)]

RF_GRID = [
    {"max_depth": d, "min_samples_leaf": leaf, "max_features": mf, "class_weight": cw}
    for d, leaf, mf, cw in product(
        (3, 5, 7, 10, None),       # tree depth
        (1, 10, 30, 100),          # min matches per leaf (larger = smoother probabilities)
        ("sqrt", 0.5, 1.0),        # share of features tried at each split
        (None, "balanced"),        # balanced up-weights draws/away wins
    )
]
RF_N_ESTIMATORS = 300


def make_model(name, params):
    if name == "LR":
        # multinomial (lbfgs default for >2 classes), L2 penalty; scaler fitted on train only inside the pipeline
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=5000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=SEED, n_jobs=-1, **params)
    raise ValueError(name)


GRIDS = {"LR": LR_GRID, "RF": RF_GRID}


def n_features_per_split(max_features, n):
    """How many features sklearn's RF actually tries per split (to spot duplicate grid points)."""
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n)))
    return max(1, int(max_features * n))


# ----------------------------------------------------------------------------- metrics (definitions in step3_modelling.md)
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


# ----------------------------------------------------------------------------- data
df = pd.read_csv(ROOT / "data_processed" / "epl_step2.csv")
splits = {s: df[df["Split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
all_features = sorted({f for fs in FEATURE_SETS.values() for f in fs})
for s, d in splits.items():
    missing = d[all_features].isna().sum()
    assert missing.sum() == 0, f"missing values in {s}: {missing[missing > 0].to_dict()}"

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


log("# Step 3 - Logistic regression + Random forest (Dickon)\n")
log("Rows: " + ", ".join(f"{s} {len(d)}" for s, d in splits.items()))
log(f"Classes order: {CLASSES}. Selection metric: val log loss. Test evaluated once, no refit.\n")

# ----------------------------------------------------------------------------- predicted league tables
def season_points(matches, P):
    """Points per (Season, Team). With P = [pH, pD, pA] these are EXPECTED points (3*P(win) + 1*P(draw));
    with P = one-hot actual results they are the real league points."""
    home = pd.DataFrame({"Season": matches["Season"], "Team": matches["HomeTeam"], "pts": 3 * P[:, 0] + P[:, 1]})
    away = pd.DataFrame({"Season": matches["Season"], "Team": matches["AwayTeam"], "pts": 3 * P[:, 2] + P[:, 1]})
    return pd.concat([home, away]).groupby(["Season", "Team"])["pts"].sum()


def season_label(start_year):
    return f"{start_year}_{(start_year + 1) % 100:02d}"  # 2024 -> "2024_25"


test_onehot = (splits["test"][TARGET].values[:, None] == np.array(CLASSES)[None, :]).astype(float)
league_tables = {"actual_pts": season_points(splits["test"], test_onehot)}

# ----------------------------------------------------------------------------- tune on val, evaluate on test
tuning_rows, result_rows = [], []
for model_name, (fs_name, feats) in product(GRIDS, FEATURE_SETS.items()):
    tag = f"{model_name}-{fs_name}"
    Xtr, ytr = splits["train"][feats], splits["train"][TARGET]
    Xva, yva = splits["val"][feats], splits["val"][TARGET]
    Xte, yte = splits["test"][feats], splits["test"][TARGET]

    t0 = time.time()
    best = None
    seen = set()
    for params in GRIDS[model_name]:
        # with few features (market set = 3) different max_features values give identical models; skip duplicates
        key_params = dict(params)
        if model_name == "RF":
            key_params["max_features"] = n_features_per_split(params["max_features"], len(feats))
        key = json.dumps(key_params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)

        m = make_model(model_name, params).fit(Xtr, ytr)
        val = metrics(yva, proba_hda(m, Xva))
        tuning_rows.append({"model": model_name, "feature_set": fs_name, "params": json.dumps(params),
                            **{f"val_{k}": v for k, v in val.items()}})
        if best is None or val["log_loss"] < best[1]["log_loss"]:
            best = (params, val, m)

    params, val, m = best
    P_test = proba_hda(m, Xte)
    test = metrics(yte, P_test)

    xpts = season_points(splits["test"], P_test)
    league_tables[f"xPts_{model_name}_{fs_name}"] = xpts
    champions = {f"champion_{season_label(s)}": g.idxmax()[1] for s, g in xpts.groupby(level="Season")}

    result_rows.append({"model": model_name, "feature_set": fs_name, "owner": OWNER,
                        **{k: round(v, 4) for k, v in test.items()}, **champions, "best_params": json.dumps(params)})
    pd.DataFrame({"match_id": splits["test"]["match_id"], "Season": splits["test"]["Season"],
                  "HomeTeam": splits["test"]["HomeTeam"], "AwayTeam": splits["test"]["AwayTeam"],
                  "FTResult": yte, "pH": P_test[:, 0], "pD": P_test[:, 1], "pA": P_test[:, 2]}
                 ).to_csv(ROOT / "results" / f"preds_{model_name}_{fs_name}.csv", index=False)

    cm = pd.DataFrame(confusion_matrix(yte, np.array(CLASSES)[P_test.argmax(1)], labels=CLASSES),
                      index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    log(f"## {tag}  ({len(seen)} candidates, {time.time() - t0:.0f}s)")
    log(f"best params: {params}")
    log("val : " + "  ".join(f"{k} {v:.4f}" for k, v in val.items()))
    log("test: " + "  ".join(f"{k} {v:.4f}" for k, v in test.items()))
    log("test confusion matrix:\n```\n" + cm.to_string() + "\n```")
    if model_name == "LR":
        coef = pd.DataFrame(m[-1].coef_, index=m[-1].classes_, columns=feats).loc[CLASSES].T.round(3)
        log("LR coefficients (standardised features; EloDiff/ProbHome collinear in `combined` - don't interpret those two):"
            "\n```\n" + coef.to_string() + "\n```")
    else:
        imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False).round(3)
        log("RF impurity importances:\n```\n" + imp.to_string() + "\n```")
    log()

# ----------------------------------------------------------------------------- save
pd.DataFrame(tuning_rows).to_csv(ROOT / "logs" / "step3_lr_rf_tuning.csv", index=False)

out = ROOT / "results" / "test_metrics.csv"
new = pd.DataFrame(result_rows)
if out.exists():  # keep other people's rows (e.g. Faris's KNN/baselines), replace only ours
    old = pd.read_csv(out)
    new = pd.concat([old[old["owner"] != OWNER], new], ignore_index=True)
new.to_csv(out, index=False)

tables = pd.DataFrame(league_tables).round(1).reset_index()
for col in [c for c in tables.columns if c.endswith("pts") or c.startswith("xPts")]:
    rank_col = "actual_rank" if col == "actual_pts" else col.replace("xPts", "rank")
    tables[rank_col] = tables.groupby("Season")[col].rank(ascending=False, method="min").astype(int)
tables = tables.sort_values(["Season", "actual_rank"])
tables.to_csv(ROOT / "results" / "league_tables_lr_rf.csv", index=False)

actual_champs = {f"champion_{season_label(s)}": g.loc[g["actual_rank"] == 1, "Team"].iloc[0]
                 for s, g in tables.groupby("Season")}
log("## Predicted champions (most expected points over the season)")
log("Actual: " + ", ".join(f"{k[9:]} {v}" for k, v in actual_champs.items()))
log("```\n" + pd.DataFrame(result_rows)[["model", "feature_set", *actual_champs]].to_string(index=False) + "\n```\n")

log("## Test summary\n```\n" + pd.DataFrame(result_rows).drop(columns="best_params").to_string(index=False) + "\n```")
(ROOT / "logs" / "step3_lr_rf_run_log.md").write_text("\n".join(log_lines), encoding="utf-8")
