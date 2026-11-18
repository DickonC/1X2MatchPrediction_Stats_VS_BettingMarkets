"""
Step 3 extra - Betting simulation on the test seasons (2024/25 + 2025/26).

Reads every results/preds_<model>_<feature_set>.csv (Dickon's LR/RF and, once they exist, Faris's KNN files),
joins the pre-match odds and simulates flat £1 stakes. Probabilities are NOT re-estimated: nothing is retrained.

Strategies (at most one bet per match):
  favourite : bet the outcome the model thinks is most likely
  value     : bet only where the model's expected value is positive, model_prob * odds > 1
              (if several outcomes qualify, take the one with the highest expected value)
Benchmarks:
  bookmaker_favourite : bet the outcome with the lowest Bet365 odds
  always_home         : bet the home team every match

Odds sources:
  b365 : Bet365 odds (main result; same bookmaker as our ProbHome/ProbDraw features)
  max  : best odds across bookmakers (side result: what "shopping around" for prices does)

Profit per bet: odds - 1 if it wins, -1 if it loses. ROI = total profit / total staked.
95% CI for ROI: bootstrap over matches (2,000 resamples, seed 42).

Outputs  results/betting_results.csv     one row per model x feature set x strategy x odds x season (+ "all")
         logs/step3_betting_run_log.md   readable tables

Usage:  python src/step3_betting.py     (run after the model scripts)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feature_sets import CLASSES, TARGET  # noqa: E402

SEED, N_BOOT = 42, 2000
ODDS = {"b365": ["OddHome", "OddDraw", "OddAway"], "max": ["MaxHome", "MaxDraw", "MaxAway"]}  # H, D, A order

data = pd.read_csv(ROOT / "data_processed" / "epl_step2.csv")
test = data[data["Split"] == "test"].set_index("match_id")
assert test[[c for cols in ODDS.values() for c in cols]].notna().all().all(), "missing odds in test"
y = test[TARGET].to_numpy()
seasons = test["Season"].to_numpy()
won = y[:, None] == np.array(CLASSES)[None, :]                       # (n, 3) bool, H D A


def bets_from_probs(P, odds, strategy):
    """Return the index (0=H, 1=D, 2=A) bet on in each match, or -1 for no bet."""
    if strategy == "favourite":
        return P.argmax(1)
    if strategy == "value":
        ev = P * odds                                                # expected return per £1
        pick = ev.argmax(1)
        return np.where(ev.max(1) > 1, pick, -1)
    raise ValueError(strategy)


def summarise(pick, odds, mask, rng):
    """Flat £1 stakes on `pick` for the matches in `mask`."""
    p, o, w = pick[mask], odds[mask], won[mask]
    bet = p >= 0
    rows = np.arange(len(p))
    profit = np.where(bet, np.where(w[rows, p.clip(0)], o[rows, p.clip(0)] - 1, -1.0), 0.0)
    n_bets, total = int(bet.sum()), float(profit.sum())
    out = {"matches": int(mask.sum()), "bets": n_bets,
           "bets_H": int((p == 0).sum()), "bets_D": int((p == 1).sum()), "bets_A": int((p == 2).sum()),
           "win_rate": (w[rows, p.clip(0)][bet].mean() if n_bets else np.nan),
           "avg_odds": (o[rows, p.clip(0)][bet].mean() if n_bets else np.nan),
           "profit": total, "roi": (total / n_bets if n_bets else np.nan)}
    # bootstrap CI for ROI: resample matches (incl. no-bet matches, so the number of bets varies too)
    idx = rng.integers(0, len(p), size=(N_BOOT, len(p)))
    boot_profit, boot_bets = profit[idx].sum(1), bet[idx].sum(1)
    boot_roi = boot_profit[boot_bets > 0] / boot_bets[boot_bets > 0]
    out["roi_ci_low"], out["roi_ci_high"] = (np.percentile(boot_roi, [2.5, 97.5]) if len(boot_roi) else (np.nan, np.nan))
    return out


# ----------------------------------------------------------------------------- collect "bettors"
bettors = []  # (model, feature_set, strategy, odds_source, pick array)
for f in sorted((ROOT / "results").glob("preds_*.csv")):
    model, fs = f.stem[len("preds_"):].split("_", 1)
    preds = pd.read_csv(f).set_index("match_id").loc[test.index]
    P = preds[["pH", "pD", "pA"]].to_numpy()
    for src, cols in ODDS.items():
        odds = test[cols].to_numpy()
        for strategy in ("favourite", "value"):
            bettors.append((model, fs, strategy, src, bets_from_probs(P, odds, strategy)))

for src, cols in ODDS.items():
    odds = test[cols].to_numpy()
    bettors.append(("Benchmark", "-", "bookmaker_favourite", src, test[ODDS["b365"]].to_numpy().argmin(1)))
    bettors.append(("Benchmark", "-", "always_home", src, np.zeros(len(test), dtype=int)))

# ----------------------------------------------------------------------------- simulate
rows = []
for model, fs, strategy, src, pick in bettors:
    odds = test[ODDS[src]].to_numpy()
    rng = np.random.default_rng(SEED)
    for season in [*sorted(set(seasons)), "all"]:
        mask = np.ones(len(test), bool) if season == "all" else seasons == season
        label = "all" if season == "all" else f"{season}/{(season + 1) % 100:02d}"
        rows.append({"model": model, "feature_set": fs, "strategy": strategy, "odds": src, "season": label,
                     **summarise(pick, odds, mask, rng)})

res = pd.DataFrame(rows)
num = ["win_rate", "avg_odds", "profit", "roi", "roi_ci_low", "roi_ci_high"]
res[num] = res[num].round(4)
res.to_csv(ROOT / "results" / "betting_results.csv", index=False)

# ----------------------------------------------------------------------------- readable log
lines = ["# Step 3 - Betting simulation (flat £1 stakes, test seasons)\n",
         f"Matches: {len(test)}. Mean Bet365 margin on test: {test['BookmakerMargin'].mean():.3f}.",
         "ROI = profit / staked. CI = 95% bootstrap over matches. Draw bets in bets_D.\n"]
show = ["model", "feature_set", "strategy", "bets", "bets_D", "win_rate", "avg_odds", "profit", "roi", "roi_ci_low", "roi_ci_high"]
for src in ODDS:
    for season in res["season"].unique():
        t = res[(res.odds == src) & (res.season == season)].sort_values(["strategy", "roi"], ascending=[True, False])
        lines += [f"## Odds: {src} · season: {season}", "```", t[show].to_string(index=False), "```", ""]
(ROOT / "logs" / "step3_betting_run_log.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:3] + lines[3:3 + 4 + len(res[(res.odds == 'b365') & (res.season == 'all')])]))
