"""Final feature sets for Step 3 (modelling). Single source of truth: import these, don't retype them.

Agreed 2026-11-17 (Dickon + Faris): start from the Step-2 lists in data_processed/features_step2.json
and add EloDiff back into the combined set (Step 2's VIF pruning had removed it from AM because of its
correlation with ProbHome). Result: 9 non-market + 3 market = 12 features in total.

    from feature_sets import FEATURE_SETS, TARGET, CLASSES
"""

NON_MARKET = [            # = features_step2.json["A"]  (9)
    "AwayElo",
    "EloDiff",
    "Form5Diff",
    "Home_Avg5_ShotsFor",
    "Home_Avg5_ShotsAgainst",
    "Home_Avg5_TargetFor",
    "Away_Avg5_ShotsFor",
    "Away_Avg5_ShotsAgainst",
    "Home_H2H_GDPG",
]

MARKET = [                # market features that survived Step 2 (3)
    "ProbHome",
    "ProbDraw",
    "MaxOver25",
]

COMBINED = NON_MARKET + MARKET   # 12 (= features_step2.json["AM"] + EloDiff)

FEATURE_SETS = {
    "non_market": NON_MARKET,
    "market": MARKET,
    "combined": COMBINED,
}

TARGET = "FTResult"
CLASSES = ["H", "D", "A"]        # fixed class/probability-column order for every model and metric

# Bookmaker baseline probabilities (Bet365, margin removed, sum to 1) - used by the baseline, NOT as model inputs
BOOKMAKER_PROBS = ["ProbHome", "ProbDraw", "ProbAway"]

assert len(NON_MARKET) == 9 and len(MARKET) == 3 and len(set(COMBINED)) == 12
