"""
Step 1 - Data preparation & cleaning (Premier League, pre-match features only).

Raw files in data_raw/ are never modified. Running this script rebuilds everything
in data_processed/ and logs/ from scratch, so every change is reproducible and
reversible (edit CONFIG or delete a fix and re-run).

Outputs
  data_processed/epl_step1.csv        one row per EPL match, target + pre-match features + Split
  logs/step1_cell_changes.csv         every individual value we corrected (old -> new, reason)
  logs/step1_dropped_rows.csv         every EPL match we removed, with reason
  logs/step1_dropped_columns.csv      every raw column we removed, with reason
  logs/step1_run_log.md               counts/diagnostics from the run

Usage:  python src/step1_prepare.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "data"
OUT = ROOT / "data_processed"
LOGS = ROOT / "logs"

# ----------------------------------------------------------------------------- CONFIG
CONFIG = {
    "division": "E0",                # Premier League only (for now)
    "season_start_month": 8,         # Aug-Jul seasons (handles the COVID 2019/20 season ending in July)
    "rolling_window": 5,             # "last N matches" window for lagged in-match stats
    "h2h_window": 5,                 # last N head-to-head meetings
    "burn_in_matches": 5,            # drop matches where either team has < N prior matches in the dataset
    "splits": {                      # by season start year (2022 = 2022/23)
        "train": (2005, 2022),       # from 2005/06: first season with complete odds (decided 2026-10-23)
        "val": (2023, 2023),
        "test": (2024, 2025),
    },
    # Same club appearing under two names (found in data checks). Applied to BOTH files.
    "team_name_map": {
        "Nott'm Forest": "Nottm Forest",   # renamed from 2024-12-26 in Matches and EloRatings
        "M'gladbach": "MGladbach",
        "Ham-Kam": "HamKam",
        "Colon Santa FE": "Colon Santa Fe",
    },
}

# Columns only known after kick-off -> leakage for a pre-match model.
POST_MATCH_COLS = {
    "FTHome": "full-time goals (target info)", "FTAway": "full-time goals (target info)",
    "HTHome": "half-time score", "HTAway": "half-time score", "HTResult": "half-time result",
    "HomeShots": "in-match stat", "AwayShots": "in-match stat",
    "HomeTarget": "in-match stat", "AwayTarget": "in-match stat",
    "HomeFouls": "in-match stat", "AwayFouls": "in-match stat",
    "HomeCorners": "in-match stat", "AwayCorners": "in-match stat",
    "HomeYellow": "in-match stat", "AwayYellow": "in-match stat",
    "HomeRed": "in-match stat", "AwayRed": "in-match stat",
    "C_LTH": "cluster prob built from in-match stats", "C_LTA": "cluster prob built from in-match stats",
    "C_VHD": "cluster prob built from in-match stats", "C_VAD": "cluster prob built from in-match stats",
    "C_HTB": "cluster prob built from in-match stats", "C_PHB": "cluster prob built from in-match stats",
}
OTHER_DROP_COLS = {
    "MatchTime": "73% missing in EPL and not a planned feature",
    "Division": "constant after filtering to E0",
}

cell_changes, dropped_rows, log_lines = [], [], []


def log(msg):
    print(msg)
    log_lines.append(msg)


def record_changes(df, mask, col, new_values, reason, table):
    """Overwrite df.loc[mask, col] and log each changed cell."""
    old = df.loc[mask, col]
    new = pd.Series(new_values, index=old.index)
    changed = ~((old == new) | (old.isna() & new.isna()))
    for idx in old.index[changed]:
        cell_changes.append({"table": table, "row_id": idx, "column": col,
                             "old": old[idx], "new": new[idx], "reason": reason})
    df.loc[old.index[changed], col] = new[changed]
    return int(changed.sum())


def season_of(dates):
    m = CONFIG["season_start_month"]
    return np.where(dates.dt.month >= m, dates.dt.year, dates.dt.year - 1)


# ----------------------------------------------------------------------------- 1. load
matches = pd.read_csv(RAW / "Matches.csv", low_memory=False, parse_dates=["MatchDate"])
elo = pd.read_csv(RAW / "EloRatings.csv", parse_dates=["date"])
matches.index.name = "match_id"  # = row number in raw Matches.csv (0-based), lets us join back to raw
elo.index.name = "elo_row_id"
log(f"Loaded Matches {matches.shape}, EloRatings {elo.shape}")

# ----------------------------------------------------------------------------- 2. team names
for col in ["HomeTeam", "AwayTeam"]:
    stripped = matches[col].str.strip()
    n = record_changes(matches, matches.index, col, stripped, "trailing whitespace in team name", "Matches")
    log(f"{col}: stripped whitespace in {n} cells")
    mask = matches[col].isin(CONFIG["team_name_map"])
    n = record_changes(matches, mask, col, matches.loc[mask, col].map(CONFIG["team_name_map"]),
                       "same club under two names", "Matches")
    log(f"{col}: unified {n} team-name cells")

# Elo: after the Forest rename, the source kept a frozen copy of the OLD name (stuck at 1731.50
# from 2025-06-15 onwards) alongside the real NEW-name series. Drop the frozen old-name rows
# that clash with a new-name row on the same date, then unify the name.
elo["_canon"] = elo["club"].replace(CONFIG["team_name_map"])
dup = elo.duplicated(["_canon", "date"], keep=False)
ghost = dup & ~elo["club"].isin(CONFIG["team_name_map"])  # the clashing row already under the canonical name
ghost_rows = elo[ghost]
LOGS.mkdir(exist_ok=True)
log(f"EloRatings: dropping {len(ghost_rows)} duplicate snapshot rows "
    f"({ghost_rows['club'].unique().tolist()}, {ghost_rows.date.min().date()} to {ghost_rows.date.max().date()})")
ghost_rows.drop(columns="_canon").to_csv(LOGS / "step1_dropped_elo_rows.csv")
elo = elo[~ghost]
n = (elo["club"] != elo["_canon"]).sum()
for idx in elo.index[elo["club"] != elo["_canon"]]:
    cell_changes.append({"table": "EloRatings", "row_id": idx, "column": "club",
                         "old": elo.at[idx, "club"], "new": elo.at[idx, "_canon"],
                         "reason": "same club under two names"})
elo = elo.assign(club=elo["_canon"]).drop(columns="_canon").sort_values("date")
log(f"EloRatings: unified {n} club-name cells")

# ----------------------------------------------------------------------------- 3. season
matches["Season"] = season_of(matches["MatchDate"])

# ----------------------------------------------------------------------------- 4. long (team-match) table
# One row per team per match across ALL divisions, so promoted teams keep their Championship history.
pts_home = matches["FTResult"].map({"H": 3, "D": 1, "A": 0})
pts_away = matches["FTResult"].map({"H": 0, "D": 1, "A": 3})
stat_pairs = {"Goals": ("FTHome", "FTAway"), "Shots": ("HomeShots", "AwayShots"),
              "Target": ("HomeTarget", "AwayTarget"), "Corners": ("HomeCorners", "AwayCorners"),
              "Fouls": ("HomeFouls", "AwayFouls"), "Yellow": ("HomeYellow", "AwayYellow"),
              "Red": ("HomeRed", "AwayRed")}


def side_frame(side):
    is_home = side == "H"
    d = pd.DataFrame({
        "match_id": matches.index, "date": matches["MatchDate"], "side": side,
        "team": matches["HomeTeam" if is_home else "AwayTeam"],
        "opp": matches["AwayTeam" if is_home else "HomeTeam"],
        "pts": pts_home if is_home else pts_away,
    })
    for name, (h, a) in stat_pairs.items():
        d[f"{name}For"] = matches[h] if is_home else matches[a]
        d[f"{name}Against"] = matches[a] if is_home else matches[h]
    return d


L = pd.concat([side_frame("H"), side_frame("A")], ignore_index=True)
L = L.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
g = L.groupby("team", sort=False)
L["n_prior"] = g.cumcount()  # matches this team has played earlier in the dataset

# Form (points in last 3 / 5), same definition as the provided columns (full history, no season reset)
for k in (3, 5):
    L[f"Form{k}"] = g["pts"].transform(lambda s: s.shift().rolling(k, min_periods=1).sum()).fillna(0)

# Lagged in-match stats: mean over the team's previous N matches (current match excluded via shift).
# min_periods=1 + NaN-skipping: averages over whatever earlier matches have that stat recorded.
W = CONFIG["rolling_window"]
roll_cols = [c for c in L.columns if c.endswith(("For", "Against"))]
for c in roll_cols:
    L[f"Avg{W}_{c}"] = g[c].transform(lambda s: s.shift().rolling(W, min_periods=1).mean())

# Head-to-head: from this team's perspective, over the previous N meetings with this opponent (any venue/division)
H = CONFIG["h2h_window"]
L["gd"] = L["GoalsFor"] - L["GoalsAgainst"]
L = L.sort_values(["team", "opp", "date", "match_id"])
gp = L.groupby(["team", "opp"], sort=False)
L["H2H_n"] = gp.cumcount().clip(upper=H)
L["H2H_PPG"] = gp["pts"].transform(lambda s: s.shift().rolling(H, min_periods=1).mean())
L["H2H_GDPG"] = gp["gd"].transform(lambda s: s.shift().rolling(H, min_periods=1).mean())

feat_cols = ["n_prior", "Form3", "Form5"] + [f"Avg{W}_{c}" for c in roll_cols] + ["H2H_n", "H2H_PPG", "H2H_GDPG"]


def side_features(side, prefix):
    s = L[L.side == side].set_index("match_id")[feat_cols]
    return s.add_prefix(prefix)


team_feats = side_features("H", "Home_").join(side_features("A", "Away_"))
# H2H is symmetric: the away-team versions are exact mirrors of the home-team ones
# (same count, GD negated, PPG = 3 - home PPG - draw share), so keep the home perspective only.
H2H_MIRRORS = {"Away_H2H_n": "identical to Home_H2H_n", "Away_H2H_GDPG": "= -Home_H2H_GDPG",
               "Away_H2H_PPG": "mirror of Home_H2H_PPG"}
team_feats = team_feats.drop(columns=list(H2H_MIRRORS))

# ----------------------------------------------------------------------------- 5. filter to EPL
epl = matches[matches["Division"] == CONFIG["division"]].copy()
log(f"\nEPL matches: {len(epl)} ({epl.Season.min()}/{epl.Season.min()+1} to {epl.Season.max()}/{epl.Season.max()+1})")
log("EPL matches per season with != 380 games: "
    + str({f"{s}/{s+1}": int(n) for s, n in epl.groupby("Season").size().items() if n != 380}))
epl = epl.join(team_feats)

# ----------------------------------------------------------------------------- 6. fix Form using corrected names
# The provided Form columns agree with our recomputation except (a) the first weeks of 2000/01
# (the source used pre-2000 history we don't have -> removed by burn-in below) and (b) matches
# broken by the Forest rename. Overwrite only outside the burn-in so we touch as little as possible.
outside_burn = (epl["Home_n_prior"] >= CONFIG["burn_in_matches"]) & (epl["Away_n_prior"] >= CONFIG["burn_in_matches"])
for side in ("Home", "Away"):
    for k in (3, 5):
        n = record_changes(epl, outside_burn, f"Form{k}{side}", epl.loc[outside_burn, f"{side}_Form{k}"],
                           "recomputed after team-name fix (raw value broken by Forest rename)", "Matches")
        log(f"Form{k}{side}: corrected {n} cells")
epl = epl.drop(columns=[f"{s}_Form{k}" for s in ("Home", "Away") for k in (3, 5)])

# ----------------------------------------------------------------------------- 7. Elo: fill gaps + Elo momentum
def elo_asof(teams, dates, offset_days=0):
    """Latest Elo snapshot on/before (date - offset) for each team; mirrors how the source filled HomeElo."""
    q = pd.DataFrame({"club": teams.values, "t": (dates - pd.Timedelta(days=offset_days)).values,
                      "_i": np.arange(len(teams))}).sort_values("t")
    r = pd.merge_asof(q, elo[["date", "club", "elo"]].rename(columns={"date": "t"}),
                      on="t", by="club", direction="backward")
    return pd.Series(r.sort_values("_i")["elo"].values, index=teams.index)


for side, team_col in (("Home", "HomeTeam"), ("Away", "AwayTeam")):
    miss = epl[f"{side}Elo"].isna()
    n = record_changes(epl, miss, f"{side}Elo", elo_asof(epl.loc[miss, team_col], epl.loc[miss, "MatchDate"]),
                       "missing Elo filled from EloRatings after team-name fix", "Matches")
    log(f"{side}Elo: filled {n} missing cells")
    # Elo momentum: change vs the snapshot ~1 and ~2 months earlier (README's EloChange1/2 idea)
    for months, days in ((1, 30), (2, 60)):
        epl[f"EloChange{months}{side}"] = epl[f"{side}Elo"] - elo_asof(epl[team_col], epl["MatchDate"], days)

# ----------------------------------------------------------------------------- 8. derived pre-match features
epl["EloDiff"] = epl["HomeElo"] - epl["AwayElo"]
epl["Form3Diff"] = epl["Form3Home"] - epl["Form3Away"]
epl["Form5Diff"] = epl["Form5Home"] - epl["Form5Away"]
inv = 1 / epl[["OddHome", "OddDraw", "OddAway"]]
epl["BookmakerMargin"] = inv.sum(axis=1, min_count=3) - 1
for c, name in (("OddHome", "ProbHome"), ("OddDraw", "ProbDraw"), ("OddAway", "ProbAway")):
    epl[name] = inv[c] / inv.sum(axis=1, min_count=3)  # margin-free implied probability

# ----------------------------------------------------------------------------- 9. drop rows
def drop_rows(mask, reason):
    global epl
    for idx in epl.index[mask]:
        dropped_rows.append({"match_id": idx, "MatchDate": epl.at[idx, "MatchDate"].date(),
                             "HomeTeam": epl.at[idx, "HomeTeam"], "AwayTeam": epl.at[idx, "AwayTeam"],
                             "reason": reason})
    log(f"Dropped {int(mask.sum())} rows: {reason}")
    epl = epl[~mask]


log("")
drop_rows(~outside_burn.reindex(epl.index),
          f"burn-in: a team has < {CONFIG['burn_in_matches']} earlier matches in dataset (form/rolling stats unreliable)")
lo, hi = min(v[0] for v in CONFIG["splits"].values()), max(v[1] for v in CONFIG["splits"].values())
drop_rows(~epl["Season"].between(lo, hi), f"season outside the train/val/test range ({lo}/{lo+1} to {hi}/{hi+1})")
drop_rows(epl["FTResult"].isna(), "missing target")

# ----------------------------------------------------------------------------- 10. drop columns, split, save
dropped_cols = [{"column": c, "reason": r} for c, r in {**POST_MATCH_COLS, **OTHER_DROP_COLS}.items()]
dropped_cols += [{"column": c, "reason": f"derived column not kept: {r}"} for c, r in H2H_MIRRORS.items()]
epl = epl.drop(columns=list(POST_MATCH_COLS) + list(OTHER_DROP_COLS))
epl = epl.drop(columns=["Home_n_prior", "Away_n_prior"])

epl["Split"] = None
for name, (a, b) in CONFIG["splits"].items():
    epl.loc[epl["Season"].between(a, b), "Split"] = name

first = ["MatchDate", "Season", "Split", "HomeTeam", "AwayTeam", "FTResult"]
epl = epl[first + [c for c in epl.columns if c not in first]]

OUT.mkdir(exist_ok=True)
epl.to_csv(OUT / "epl_step1.csv")
pd.DataFrame(cell_changes).to_csv(LOGS / "step1_cell_changes.csv", index=False)
pd.DataFrame(dropped_rows).to_csv(LOGS / "step1_dropped_rows.csv", index=False)
pd.DataFrame(dropped_cols).to_csv(LOGS / "step1_dropped_columns.csv", index=False)

# ----------------------------------------------------------------------------- 11. diagnostics
log(f"\nFinal dataset: {epl.shape[0]} matches x {epl.shape[1]} columns -> data_processed/epl_step1.csv")
log("\nRows per split / class balance:")
tab = pd.crosstab(epl["Split"], epl["FTResult"], margins=True)
log(tab.to_string())
log("\n" + (pd.crosstab(epl["Split"], epl["FTResult"], normalize="index").round(3)).to_string())
na = (epl.isna().mean() * 100).round(1)
log("\nColumns with missing values (% of rows):")
log(na[na > 0].to_string())
log("\nMissing values by split for odds columns that are absent in early seasons:")
log(epl.groupby("Split")[["OddHome", "MaxHome", "Over25", "HandiSize"]].apply(lambda d: d.isna().mean().round(3)).to_string())
log(f"\nTotal logged cell changes: {len(cell_changes)}")

(LOGS / "step1_run_log.md").write_text("# Step 1 run log (auto-generated)\n\n```\n" + "\n".join(log_lines) + "\n```\n")
