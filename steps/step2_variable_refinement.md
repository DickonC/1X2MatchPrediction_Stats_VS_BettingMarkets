# Step 2 – Variable refinement

**Script:** `src/step2_refine.py` (run `python src/step2_refine.py`, ~5 min because Boruta runs with 3 seeds)
**Input:** `data_processed/epl_step1.csv`
**Outputs:**
| File | Contents |
|---|---|
| `data_processed/features_step2.json` | final feature lists `A`, `AM` (+ pre-VIF lists for tree models) |
| `data_processed/epl_step2.csv` | step-1 data + the imputations below (input for Step 3) |
| `logs/step2_feature_decisions.csv` | **every candidate feature → stage it was dropped at + reason + scores** |
| `logs/step2_imputations.csv` | every imputation (column, split, number of cells, fill value) |
| `logs/step2_run_log.md` | full printed output of the last run |
| `figures/step2/*.png` | the figures below |

> ⚠️ All selection uses **train only** (2005/06–2022/23, n = 6,840). Val/test were never looked at.

Two feature groups throughout, so we can later answer *"do pre-match features add anything beyond the bookmaker?"*:
- **A – non-market (44 candidates):** Elo, EloChange, Form, Avg5 stats, H2H
- **M – market (17 candidates):** Bet365/Max odds, implied probabilities, margin, Over/Under 2.5, Asian handicap

---

## Result in one picture

```
61 candidates (44 A + 17 M)
 │ 2.1 correlation: redundant pairs (|ρ| ≥ 0.85)          −13
48
 │ 2.2 feature vs class: no detectable difference         −9
39 (34 A + 5 M)
 │ 2.3 Boruta (3 seeds, majority vote)       A run: 34 → 10     A+M run: 39 → 13
 │ 2.4 multicollinearity (VIF < 10)                  → 9               → 11
 ▼

## Remaining Features
(Home H2H GDPG was average goal difference per game in last 5 meetings with oposition,any venue home or away) (e.g. past 5 chelsea united matches, chelsea goals - united goals)

A  (non-market, 9):  AwayElo, EloDiff, Form5Diff, Home_Avg5_ShotsFor, Home_Avg5_ShotsAgainst,
                     Home_Avg5_TargetFor, Away_Avg5_ShotsFor, Away_Avg5_ShotsAgainst, Home_H2H_GDPG


AM (+ market, 11):   A without EloDiff  +  ProbHome, ProbDraw, MaxOver25
(Bookies Probabilities effectively hold relevent info from EloDiff (High correlation))

UPDATE 2026-11-17 → Step 3 uses EloDiff back in: combined = A (9) + 3 market = 12
```

---

## What we did, step by step

### 2.0 Start from 2005/06 + imputation
- **Training now starts at 2005/06** (group decision): the first season where every odds column is recorded. This was applied in Step 1's `CONFIG`, so the 1,779 removed 2000/01–2004/05 matches are logged in `logs/step1_dropped_rows.csv`. Rolling features for 2005 still use the earlier matches as history.
- **H2H for teams that had never met** (179 train + 10 val rows): `Home_H2H_GDPG` = 0 (no evidence either way), `Home_H2H_PPG` = train mean (1.348). `Home_H2H_n = 0` already marks these rows.
- **Asian handicap missing** for 11 matches: filled with the train median.
- **Why train-only fill values:** using val/test statistics to impute would leak information from them.

### 2.1 Correlation: removing redundancy → `01_correlation_matrix.png`
- **Method:** Spearman ρ (rank-based, so robust to skewed odds). Any two features with |ρ| ≥ 0.85 count as redundant. Linked pairs are grouped into clusters, and **in each cluster we keep the member with the highest mutual information with the result**, i.e. the one that tells us most about H/D/A.
- **Clusters are formed within A and within M separately.** `EloDiff` correlates 0.94 with the bookmaker's `ProbHome`, so a single combined pass would wipe out Elo and leave nothing to test "beyond the bookmaker" with. The cross-group correlation is a result in itself: **Elo difference carries almost the same information as the odds.**

| Cluster | Kept | Dropped |
|---|---|---|
| Home-win odds (7 features, ρ up to 1.00) | `ProbHome` | `OddHome`, `MaxHome`, `OddAway`, `MaxAway`, `ProbAway`, `HandiSize` |
| Draw odds | `ProbDraw` | `OddDraw`, `MaxDraw` |
| Over/Under 2.5 odds | `MaxOver25` | `Over25`, `Under25`, `MaxUnder25` |
| Handicap price | `HandiHome` | `HandiAway` |
| Head-to-head | `Home_H2H_GDPG` | `Home_H2H_PPG` |

- Nothing within A apart from H2H crossed 0.85. Form3/Form5 (≈0.8) and shots/target/corners (≈0.6–0.75) are correlated but below the threshold, so they're left for the later stages.
- **Reading the figure:** blue = positive, red = negative. Features are ordered so correlated blocks sit together (hierarchical clustering). Annotated cells have |ρ| ≥ 0.85, and grey axis labels mark dropped features.

### 2.2 Feature vs class: relevance → `02_boxplots_by_result.png`, `03_relevance_ranking.png`
- **Boxplots** of each of the 48 remaining features split by H / D / A, sorted by effect size.
- Eyeballing alone isn't defensible, so each feature also gets two numbers:
  - **Kruskal–Wallis** test (non-parametric ANOVA) with a Bonferroni correction for 48 tests, plus the **effect size ε²**. With 6,840 rows a tiny difference is "significant", so ε² is what shows how *big* it is.
  - **Mutual information** with the result, compared with a **chance level**: the 95th percentile of MI when the result column is randomly shuffled (20 times). MI also catches non-monotonic relationships that the boxplots miss.
- **Drop rule:** drop only if **both** say nothing: KW not significant **and** MI not above chance. We wanted to be lenient here and leave the real decision to Boruta.
- **Dropped (9):** all four **Red card** averages, `Home_Avg5_YellowFor`, `Home_Avg5_FoulsAgainst`, and **Elo momentum** (`EloChange2Home`, `EloChange1Away`, `EloChange2Away`).
- **What the boxplots show:**
  - Strength features (ProbHome, EloDiff, Elo, H2H GD, Form5Diff) shift clearly from H → D → A.
  - **Draws sit in the middle** for almost every feature. The only feature where D stands out is `ProbDraw` (higher when a draw is likely) and, weakly, `MaxOver25` (low-scoring games draw more). This is why draws are hard to predict.
  - Discipline stats (cards, fouls) and recent *change* in Elo show no difference between classes.

### 2.3 Boruta: all-relevant selection → `04_boruta_A.png`, `05_boruta_AM.png`
- **How it works:** every feature gets a shuffled "shadow" copy (pure noise). A random forest is trained repeatedly, and a feature is **confirmed** only if its importance beats the *best* shadow feature significantly more often than chance.
- **Settings:** `RandomForestClassifier(class_weight="balanced", max_depth=7)`, up to 200 iterations, **3 random seeds, majority vote**. A single run flipped the borderline corner features, so we checked stability.
- **Two runs:** A only (34 features) and A+M (39 features).
- **Results:**
  - Confirmed in both: Elo (Home, Away, Diff), `Form5Diff`, `Home_H2H_GDPG`, shots for/against (home & away), `Home_Avg5_TargetFor`. The A+M run also confirmed `ProbHome`, `ProbDraw` and `MaxOver25`.
  - Rejected: all **Form3** variants (Form5 is the same information, less noisy), **goals** for/against (shots are a more stable signal of team quality than goals, in line with the expected-goals idea), corners, fouls, yellows, `BookmakerMargin`, `HandiHome`, `Home_H2H_n`.
  - Not unanimous: the corner-against features (1/3 seeds in the A run → rejected) and `Home_Avg5_TargetFor` (2/3 in A+M → confirmed).
- **Reading the figure:** each row shows the distribution of a feature's importance across Boruta iterations. The dashed line and grey band are the best shadow (noise) feature. Blue = confirmed, grey = rejected. Anything to the right of the band reliably beats noise.

### 2.4 Multicollinearity check (VIF)
- **Why an extra step:** Boruta is *all-relevant*, so it keeps correlated features if each carries signal. Pairwise correlation also misses linear combinations: **`EloDiff = HomeElo − AwayElo` exactly**, yet no pair crosses 0.85. That would make logistic regression unidentifiable.
- **Method:** variance inflation factor (VIF). While any VIF ≥ 10, drop the high-VIF feature that Boruta ranked least important.
  - **A:** `HomeElo` dropped (VIF = ∞). `AwayElo` + `EloDiff` carry exactly the same information.
  - **AM:** `HomeElo` dropped (∞), then `EloDiff` (VIF 12 with `ProbHome`). Once the odds are present, Elo difference is largely absorbed by the bookmaker's probability.
- After pruning, max VIF = 4.8 (A) and 6.1 (AM).
- Tree models don't suffer from multicollinearity, so the pre-VIF lists are also saved (`A_boruta_before_vif`, `AM_boruta_before_vif`) in case we want them for random forest / boosting.

---

## Talking points for the meeting / report
- **Elo difference ≈ the odds** (ρ = 0.94): much of what the market knows is team strength.
- **Recent form matters over 5 games, not 3**, and only as a *difference* between the teams.
- **Head-to-head goal difference survives everything**, even with Elo and odds present. This is directly relevant to coursework question 3 (recent vs historical form).
- **Shots beat goals** as a form signal; discipline stats and Elo *momentum* carry nothing.
- **Draws are barely separable** by any single feature. Expect low draw recall in Step 3 and consider class weighting / probability-based evaluation.

## Caveats
- The thresholds (|ρ| 0.85, VIF 10, α 0.05) are conventional choices, not tuned. All are in `CONFIG` for sensitivity checks.
- Keeping the highest-MI member of a cluster favours information over interpretability. In practice the kept features (`ProbHome`, `ProbDraw`, `H2H_GDPG`) are also the most interpretable.
- Boruta importance comes from one model family (random forest). A feature rejected here could still help a linear model.

## Update 2026-11-17: feature sets used in Step 3
Agreed after the progress report: **`EloDiff` is added back** to the combined set, so the market set is exactly non-market + market. Step 3 uses three sets: `non_market` (= A, 9), `market` (ProbHome, ProbDraw, MaxOver25, 3) and `combined` (12). They are defined in `src/feature_sets.py`. `features_step2.json` is left unchanged as the record of what Step 2 produced.

## Next (Step 3)
See [step3_modelling.md](step3_modelling.md). Models: LR, RF and KNN on all 3 sets. Baselines: Always-Home and the bookmaker's implied probabilities.
