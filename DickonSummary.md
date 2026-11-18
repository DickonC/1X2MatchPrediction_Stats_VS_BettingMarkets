# Dickon – Summary (LR + RF models)

Script: `src/step3_lr_rf.py` · Full logs: `logs/step3_lr_rf_run_log.md`, `logs/step3_lr_rf_tuning.csv`

---

## Results
**`results/test_metrics.csv` is where the final results can be seen.**

Test seasons 2024/25 + 2025/26 (760 matches):

| Model | Features | Accuracy | Macro F1 | Log loss | Brier | ROC-AUC |
|---|---|---|---|---|---|---|
| LR | non_market | 0.509 | 0.380 | 1.018 | 0.611 | 0.626 |
| LR | market | 0.517 | 0.387 | 1.000 | 0.598 | 0.646 |
| LR | combined | 0.509 | 0.381 | 1.005 | 0.601 | 0.642 |
| RF | non_market | 0.503 | 0.374 | 1.020 | 0.612 | 0.623 |
| RF | market | 0.504 | 0.376 | 1.007 | 0.603 | 0.646 |
| RF | combined | 0.508 | 0.381 | 1.004 | 0.602 | 0.647 |

**Predicted league champions** are also recorded in `results/test_metrics.csv` (columns `champion_2024_25`, `champion_2025_26`). Full predicted tables are in `results/league_tables_lr_rf.csv`.
- **Method:** each team's expected points for a match = 3 × P(win) + 1 × P(draw). Summed over the season, the team with the most expected points is the predicted champion.
- **Caveat:** this isn't a pre-season prediction. Each match is predicted with information available just before kick-off (Elo, form and odds update during the season).

| Model | Features | 2024/25 | 2025/26 |
|---|---|---|---|
| **Actual** | | **Liverpool** | **Arsenal** |
| LR | non_market | Liverpool ✅ | Arsenal ✅ |
| LR | market | Liverpool ✅ | Arsenal ✅ |
| LR | combined | Liverpool ✅ | Arsenal ✅ |
| RF | non_market | Man City ❌ | Arsenal ✅ |
| RF | market | Liverpool ✅ | Man City ❌ |
| RF | combined | Liverpool ✅ | Arsenal ✅ |

### Betting simulation
Script `src/step3_betting.py`. Results in `results/betting_results.csv` (per season and overall) and `logs/step3_betting_run_log.md`. A flat £1 bet per match on both test seasons, using the **Bet365 odds** from the dataset. Profit = odds − 1 if the bet wins, −1 if it loses. ROI = profit ÷ amount staked.
- **Favourite strategy:** bet the model's most likely outcome every match.
- **Value strategy:** only bet when model probability × odds > 1, i.e. the model thinks an outcome is more likely than the bookmaker's price implies.
- **Benchmarks:** always bet the bookmaker's favourite; always bet home.

Both test seasons, Bet365 odds (95% bootstrap CI):

| Model | Features | Favourite ROI | Value ROI (bets) |
|---|---|---|---|
| LR | non_market | −7.0% | −9.2% (654) |
| LR | market | −7.1% | −19.1% (300) |
| LR | combined | −8.7% | −16.3% (506) |
| RF | non_market | −9.4% | −15.8% (683) |
| RF | market | −10.3% | −15.9% (574) |
| RF | combined | −9.0% | −9.4% (594) |
| Benchmark: bookmaker favourite | | −7.4% | |
| Benchmark: always home | | −13.5% | |

**In plain words:** following our models would have lost money. The losses are **roughly the bookmaker's margin (5% on average)**, the "cut" Bet365 builds into its odds. A model that is about as accurate as the bookmaker can't overcome that. **One setup did make money:** LR non_market with the value strategy and the best available (`Max`) odds was up **£21.77 on 380 bets (+5.7% ROI) in 2024/25**. But its 95% CI runs from −10% to +24%, and the same setup lost money in 2025/26, so it was **mostly luck**, not a real edge.

**How the 95% CI is calculated (bootstrap):** we randomly resample the test matches with replacement 2,000 times. That builds 2,000 alternative test sets of the same size, and we recompute the ROI on each one. The middle 95% of those ROIs is the interval. It shows how much the ROI could swing purely from *which* matches happened to be in the test seasons.

- **Every strategy loses money.** The favourite strategy loses about 7–10%, roughly the same as just backing the bookmaker's favourite (−7.4%). Bet365's built-in margin is 5.0% on average.
- **Value betting loses more.** Where our models disagree with the bookmaker, the bookmaker is usually right. So the models can't beat the market's prices.
- **CIs are wide** (around ±7% for favourite, ±10–15% for value). With 760 matches, differences between models are mostly noise.
- **Side result: best available odds** (`Max*` columns, margin 1.6%) cut the favourite losses to about −5%. Value betting then bets on nearly every match, so it isn't meaningful there. In practice these best prices are hard to get consistently.

**The models (almost) never predicted a draw.** A draw was the most likely outcome in only 1 of 760 test matches across all 6 models (RF non_market). The predicted draw probability peaked at about 30–36%, but home or away win was always at least as likely. The bookmaker never makes a draw the favourite either (0 of 6,840 train matches).

---

## Random forest
**300 trees in every model** *(Reference)*. More trees is better, or at least never worse. It just takes longer, so it wasn't tuned.

**Tuned:**
- **`max_depth`**: maximum depth of each tree, i.e. how many questions in a row a tree can ask.
- **`min_samples_leaf`**: minimum number of matches in a leaf node. Larger values prevent overfitting.
- **`max_features`**: the share of features each split can choose from. `1.0` = all.
- **`class_weight`**: unanimously `None`. "balanced" scored worse on val log loss for every feature set.

**Method:** we tried different combinations of hyperparameters. Each was trained on the training seasons (2005/06–2022/23) and scored on 2023/24 by log loss. **We took the hyperparameters with the lowest log loss.**

| Feature set | Combinations tried |
|---|---|
| non_market | 120 |
| market | 80 |
| combined | 120 |

(Market has only 80 because with 3 features, `sqrt` and `0.5` both mean "1 feature", so those duplicates were skipped.)

### Grid of combinations
Hyperparameters were selected by **grid search over conventional ranges** (see table). Every combination of the values below was tried: 5 × 4 × 3 × 2 = 120.

| Hyperparameter | Values tried |
|---|---|
| `max_depth` | 3, 5, 7, 10, None (unlimited) |
| `min_samples_leaf` | 1, 10, 30, 100 |
| `max_features` | sqrt, 0.5, 1.0 |
| `class_weight` | None, balanced |

### Chosen hyperparameters
| Model | max_depth | min_samples_leaf | max_features | class_weight | Val log loss |
|---|---|---|---|---|---|
| RF non_market | 10 | 30 | sqrt | None | 0.9272 |
| RF market | 3 | 1 | 1.0 | None | 0.9114 |
| RF combined | 5 | 1 | 0.5 | None | 0.9118 |

---

## Logistic regression
**Tuned:** `C`, 10 values: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100. Features were standardised first.

**C = 100 means practically no restriction on the weights.** Val log loss was the same for every C ≥ 1, so regularisation made no difference. C = 100 was chosen for all 3 feature sets.

---

## Quick summary
Hyperparameters were chosen by validation log loss. Many configurations scored within ~0.001, so the specific values matter little. The consistent findings were that class weighting hurt probability quality, and that the market-based models needed only shallow trees because ProbHome carries most of the signal.
