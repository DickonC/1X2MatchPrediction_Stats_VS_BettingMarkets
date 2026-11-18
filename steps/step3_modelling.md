# Step 3 – Modelling & evaluation

**Status:** 🔄 in progress, as agreed on 2026-11-17 (Dickon + Faris), following the progress report (`AMLGroupProposal.pdf`)
**Input:** `data_processed/epl_step2.csv` (step-1 data + imputations; has a `Split` column)
**Feature lists:** `src/feature_sets.py` (import it, don't retype the lists)

---

## Question
Can pre-kick-off information predict an EPL result (**1X2 = H / D / A**), and **does adding bookmaker (market) information improve prediction** over non-market features alone?

## Data split (by season; already in the `Split` column)
| Split | Seasons | Matches | H / D / A |
|---|---|---|---|
| train | 2005/06–2022/23 | 6,840 | 46.0 / 24.3 / 29.7 % |
| val | 2023/24 | 380 | 46.1 / 21.6 / 32.4 % |
| **test** | **2024/25 + 2025/26** | **760** | 41.7 / 25.9 / 32.4 % |

The test set is ready and has never been looked at. Note that home advantage is lower in test than in train (see the proposal). We expect this to hurt Always-Home and maybe the models too.

## Feature sets (change from Step 2)
In Step 2 we ended with **A = 9 non-market** and **AM = 11** (A without `EloDiff`, plus 3 market features). **We have now added `EloDiff` back into the combined set**, which gives three sets:

| Set | n | Features |
|---|---|---|
| `non_market` | 9 | AwayElo, EloDiff, Form5Diff, Home_Avg5_ShotsFor, Home_Avg5_ShotsAgainst, Home_Avg5_TargetFor, Away_Avg5_ShotsFor, Away_Avg5_ShotsAgainst, Home_H2H_GDPG |
| `market` | 3 | ProbHome, ProbDraw, MaxOver25 |
| `combined` | 12 | non_market + market |

- **Why add EloDiff back:** the combined set should be exactly *non-market + market*. That way, comparing `combined` with `market` answers "do our features add anything beyond the bookmaker?" without a confound.
- **Consequence:** in `combined`, `EloDiff` and `ProbHome` are collinear (ρ = 0.94, VIF ≈ 12). Random forest doesn't mind. For logistic regression, use L2 regularisation (sklearn's default) and **interpret its coefficients with care**: this collinearity only affects coefficient stability, not predictive performance. Mention it in the report.

## Models: 3 models × 3 feature sets = 9, plus 2 baselines

| | `non_market` | `market` | `combined` | Owner |
|---|---|---|---|---|
| Logistic regression | LR-N | LR-M | LR-C | **Dickon** |
| Random forest | RF-N | RF-M | RF-C | **Dickon** |
| K-nearest neighbours | KNN-N | KNN-M | KNN-C | Faris |
| Baseline: Always-Home | — | — | — | Faris |
| Baseline: Bookmaker (ProbHome/Draw/Away, margin removed) | — | — | — | Faris |

Notes per model:
- **LR:** multinomial, features **standardised** (scaler fitted on the training data only, inside a `Pipeline`). Tune `C`.
- **RF:** no scaling needed. Tune `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Consider `class_weight="balanced"` (draws, see below).
- **KNN:** standardise (distance-based). Tune `k` and the weights.
- **Draws:** Step 2 showed draws sit "in the middle" of every feature. Expect low draw recall. Macro F1 will expose this, so report the confusion matrix too.

## Protocol (same for everyone, so results are comparable)
1. **Train** each candidate hyperparameter setting on *train* (2005/06–2022/23).
2. **Choose** the setting with the best **log loss on val** (2023/24). We care about probabilities, not just the predicted class.
3. **Evaluate once on test** (2024/25 + 2025/26) with **that same model**, trained on train only. **No refit on train + val** (decided 2026-11-17). No going back to tune after looking at test.
4. Fix `random_state = 42` wherever randomness is involved.

> In short: **train = learn, val = pick settings, test = final score (used once).**

## Metrics (fixed definitions, class order `["H","D","A"]`)
| Metric | Definition | Better |
|---|---|---|
| Accuracy | argmax of predicted probabilities vs true result | ↑ |
| Macro F1 | F1 per class (H, D, A), unweighted mean | ↑ |
| Log loss | mean of −ln(p of the actual result), natural log | ↓ |
| Multiclass Brier | mean over matches of Σ_k (p_k − y_k)², **range 0–2** (sum over classes, not divided by 3) | ↓ |
| ROC-AUC | one-vs-rest, **macro** average (mean of the 3 binary AUCs: H vs rest, D vs rest, A vs rest) | ↑ |

⚠️ **Column-order trap:** sklearn orders `predict_proba` columns alphabetically (`A, D, H`), and `log_loss` / `roc_auc_score` **also assume that sorted order**. Passing H, D, A columns with `labels=["H","D","A"]` gives a **silently wrong log loss** and a `ValueError` for ROC-AUC. **Easiest fix: copy the `metrics()` and `proba_hda()` functions from `src/step3_lr_rf.py`.** They store probabilities as H, D, A and compute every metric explicitly, with a built-in check against sklearn.

**Baselines' probabilities:** Always-Home predicts H for every match (accuracy, F1). A hard [1,0,0] gives infinite log loss, so it needs some probabilities for the probability metrics. **Faris decides how.** One option is the train class frequencies (H .460 / D .243 / A .297), which makes ROC-AUC 0.5 by construction. Bookmaker: probabilities = `ProbHome/ProbDraw/ProbAway`, prediction = argmax.

## Recording results
Each person writes their test results to `results/` (one row per model × feature set):

`results/test_metrics.csv` columns: `model, feature_set, owner, accuracy, macro_f1, log_loss, brier, roc_auc, best_params, champion_2024_25, champion_2025_26` (champion = team with most expected points, 3·P(win) + P(draw), over the season; full tables in `results/league_tables_lr_rf.csv`)

Also save per-match test probabilities (`results/preds_<model>_<feature_set>.csv`: `match_id, Season, FTResult, pH, pD, pA`). That way any metric, the confusion matrix or the per-season accuracy (drift check) can be recomputed without retraining.

Scripts: `src/step3_lr_rf.py` (Dickon), `src/step3_knn_baselines.py` (Faris).

## Dickon's checklist
- [ ] LR-N · LR-M · LR-C: fit on train → pick settings on val → test metrics
- [ ] RF-N · RF-M · RF-C: fit on train → pick settings on val → test metrics
- [ ] Write 6 rows to `results/test_metrics.csv` + 6 prediction files
- [ ] Record chosen hyperparameters + val scores: `logs/step3_lr_rf_run_log.md` + `logs/step3_lr_rf_tuning.csv` (written by `src/step3_lr_rf.py`)

## Notes for the report
- **EloDiff and ProbHome are collinear in the `combined` set** (Spearman ρ = 0.94, VIF ≈ 12). That's why Step 2's VIF step had removed EloDiff, and we deliberately added it back so that `combined` = `non_market` + `market` exactly.
  - **Random forest:** unaffected. Trees just pick one of the two at each split. Feature importance gets *shared* between them, so neither looks as important as it really is.
  - **Logistic regression:** predictions, and therefore all our metrics, are fine. The **individual coefficients** of EloDiff and ProbHome are unstable and can't be read as "the effect of EloDiff holding ProbHome fixed". Tuning chose `C = 100`, but val log loss is flat for every C ≥ 1 (differences < 0.0001), so in practice LR is **unregularised**. Regularisation neither helps nor hurts here.
  - **In the report:** state this caveat, and don't draw conclusions from LR-C coefficients or RF-C importances for these two features. Compare *performance* across feature sets instead.
- **Elo difference ≈ the odds** (ρ = 0.94) is itself a talking point: much of what the market knows is team strength.

## Other group work
UMAP visualisation and accuracy per test season (2024/25 vs 2025/26, drift in home advantage): **Joseph**. The per-match prediction files in `results/` are enough for the per-season accuracy.

## Open questions
See "Open questions" in `plan.md`.

## Extra: betting simulation (`src/step3_betting.py`)
Reads **every** `results/preds_*.csv`, so Faris's KNN files are included automatically once they exist (just re-run it). Flat £1 stake per match on test. Two strategies: **favourite** (bet the model's most likely outcome) and **value** (bet only if model prob × odds > 1). Benchmarks: bookmaker favourite, always home. Odds: Bet365 (main) and best-available `Max*` (side result). Outputs `results/betting_results.csv` (ROI with 95% bootstrap CI, per season + overall) and `logs/step3_betting_run_log.md`.
