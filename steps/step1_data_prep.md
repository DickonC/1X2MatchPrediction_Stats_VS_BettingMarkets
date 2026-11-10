# Step 1 – Data preparation & cleaning

**Script:** `src/step1_prepare.py` (run `python src/step1_prepare.py`)
**Output:** `data_processed/epl_step1.csv` with **7,980 EPL matches × 67 columns** (2005/06–2025/26)

## Reversibility
- `data_raw/` is never modified. The script rebuilds everything from it, so any decision can be undone by editing `CONFIG` at the top of the script and re-running.
- `match_id` (first column of the output) = row number in raw `Matches.csv`, so every row joins back to the raw data.
- Every change is logged in `logs/`:

| File | Contents |
|---|---|
| `step1_cell_changes.csv` | every individual value changed (table, row, column, old → new, reason). 806 cells |
| `step1_dropped_rows.csv` | every EPL match removed + reason (1,850) |
| `step1_dropped_columns.csv` | every column removed + reason |
| `step1_dropped_elo_rows.csv` | duplicate Elo snapshots removed (60) |
| `step1_run_log.md` | counts, class balance, missing values from the last run |

---

## What we did (in order) and why

### 1. Fixed team names (both files)
- Stripped trailing spaces (43 cells, non-EPL teams like `"Ajax "`).
- Unified clubs that appear under two names: **`Nott'm Forest` → `Nottm Forest`** (the source renamed Forest on 2024-12-26), plus `M'gladbach`, `Ham-Kam` and `Colon Santa FE` (non-EPL).
- **Why:** all per-team history (form, rolling stats, head-to-head, Elo lookup) is grouped by team name. The rename split Forest into two "teams", which caused 2024/25 to show 21 teams and broke the next three items.

### 2. Removed duplicate Elo snapshots
- After the rename, `EloRatings.csv` contains a **frozen copy** of the old Forest name stuck at 1731.50 from 2025-06-15 onwards, alongside the real series. The same happens for MGladbach. We dropped the 60 frozen rows.
- **Why:** otherwise Forest's Elo in 2025/26 (part of our test set) would be wrong.

### 3. Defined `Season` with an **August** cutoff
- **Why:** a July cutoff wrongly assigns the July 2020 COVID restart games to 2020/21. With August, every season has 380 games, except 2003/04 and 2004/05 (335 each: missing from the source, can't be fixed without outside data) and 2026/27 (in progress).

### 4. Built pre-match features from each team's **history across all divisions**
A long table with one row per team per match. Each feature uses only matches **strictly before** the current one (`shift()` before rolling).
- **Lagged in-match stats** (your idea): mean over the team's **last 5 matches** of goals, shots, shots on target, corners, fouls, yellows, reds, each *for* and *against* → `Home_Avg5_*`, `Away_Avg5_*` (28 columns).
- **Head-to-head:** over the last 5 meetings of the two clubs (any venue), from the home team's perspective: `Home_H2H_n` (number of meetings, max 5), `Home_H2H_PPG` (points per game), `Home_H2H_GDPG` (goal difference per game). The away-team versions are exact mirrors, so they were not kept.
- **Why all divisions:** promoted teams keep their Championship history instead of starting from nothing (the Championship has complete match stats). This is also how the provided Form columns are calculated (full history, no reset each season).
- Window sizes are in `CONFIG` (`rolling_window`, `h2h_window`) so we can try 3/10 later.
- **Checked by hand:** Arsenal v Crystal Palace 2024-01-20. Avg shots (18.8) and H2H GD (0.6) match manual calculations from raw data, so no current-match leakage.

### 5. Corrected Form where the rename broke it
- Recomputed Form3/Form5 with the fixed names. Raw and recomputed values agree on 99.7% of rows; the only real errors were Forest's first match after the rename (Form reset to 0). **3 cells corrected.**
- The other disagreements are in the first weeks of 2000/01, where the source used pre-2000 history we don't have. Those rows are removed by the burn-in (step 8).

### 6. Elo
- **Should we combine the Elo table?** `HomeElo`/`AwayElo` in Matches **already are** the latest Elo snapshot on or before the match date (checked: identical on 98% of EPL rows; the rest are in the source's provisional 2025/26 continuation). A ClubElo rating dated *d* is the rating going into day *d*, so this is pre-match information.
- What the Elo table adds is **Elo momentum**, so we built `EloChange1Home/Away` and `EloChange2Home/Away` = current Elo − Elo ~30/60 days earlier.
- Filled the 2 EPL matches with missing Elo (Forest, right after the rename) from the Elo table.

### 7. Derived features
- `EloDiff`, `Form3Diff`, `Form5Diff`
- `ProbHome/Draw/Away`: Bet365 implied probabilities **with the bookmaker margin removed** (1/odds, renormalised to sum to 1), plus `BookmakerMargin`.
- **Why:** raw odds are skewed and the three are tied together; normalised probabilities are interpretable and are what we compare our models against.

### 8. Dropped rows (1,850)
- **51: burn-in.** Either team has fewer than 5 earlier matches in the dataset, so form and rolling stats are incomplete (start of 2000/01).
- **1,779:** 2000/01–2004/05. **Training starts at 2005/06** (group decision, 2026-10-23), the first season where every odds column (`Max*`, Over/Under, handicap) is recorded. These matches are still used as *history* for the rolling features of 2005/06.
- **20:** the 2026/27 season, which is outside the split.

### 9. Dropped columns (leakage + unused)
| Dropped | Why |
|---|---|
| `FTHome`, `FTAway` | the result itself (used only to build lagged features) |
| `HTHome`, `HTAway`, `HTResult` | half-time = during the match |
| Shots, Target, Fouls, Corners, Yellow, Red (home & away) | during the match. Replaced by their **last-5 averages** |
| `C_LTH … C_PHB` | cluster probabilities computed from in-match stats (also missing for all of 2025/26) |
| `MatchTime` | 73% missing |
| `Division` | constant (E0) |

### 10. Split column
`Split` = `train` (2005/06–2022/23, **6,840**), `val` (2023/24, **380**), `test` (2024/25–2025/26, **760**).

| | H | D | A |
|---|---|---|---|
| train | 46.0% | 24.3% | 29.7% |
| val | 46.1% | 21.6% | 32.4% |
| test | 41.7% | 25.9% | 32.4% |

Home advantage is lower in the test seasons (41.7% H), which is worth mentioning in the report as a possible distribution shift.

**Why split by season and not randomly** (decided 2026-10-23):
- **Avoids leakage across time.** Features are rolling windows over neighbouring matches, so with a random split a test match's form/Avg5 values overlap with training matches played the same week. The model would also be trained on *later* seasons and asked about *earlier* ones (e.g. learning 2024 Man City to predict 2015).
- **Matches real use.** A pundit predicts *upcoming* matches using the past only, so a chronological hold-out estimates how well the model generalises to the future.
- **Standard practice** in football/sports forecasting papers.
- For tuning within train we'll use **season-based (expanding-window) cross-validation**, not random k-fold, for the same reasons.

## Notes on columns
- `Form3Home/Away`, `Form5Home/Away` are **original dataset columns** (points from the last 3/5 matches). We only corrected 3 cells (step 5). The Form3Diff/Form5Diff columns are ours.
- `MaxHome … HandiAway` look blank at the top of the file because the file is sorted by date and these odds **weren't recorded in early seasons**: `Max*` and `Over/Under25` are missing for 2000/01–2004/05, and `Handi*` for 2000/01–2002/03. From 2005/06 onwards they're ~100% complete; no column is entirely empty. They were deliberately left as-is, with no imputation (odds can't be sensibly made up). **Decision:** train from 2005/06, where they're complete (see step 8).

---

## Issues handed to Step 2 (resolved there)
- ~~Missing odds in early seasons~~ → resolved by starting training at 2005/06 (above).
- ~~H2H missing when teams had never met~~ → imputed in Step 2.0.
- Still a known limitation: 2003/04 and 2004/05 are each missing 45 matches in the source, so rolling stats just after those gaps are slightly off (only affects early 2005/06 history).
- Still a known limitation: the rolling stats of promoted teams come from Championship games (weaker opposition), which may inflate them.
