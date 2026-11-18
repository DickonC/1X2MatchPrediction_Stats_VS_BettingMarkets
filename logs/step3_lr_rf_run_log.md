# Step 3 - Logistic regression + Random forest (Dickon)

Rows: train 6840, val 380, test 760
Classes order: ['H', 'D', 'A']. Selection metric: val log loss. Test evaluated once, no refit.

## LR-non_market  (10 candidates, 0s)
best params: {'C': 100}
val : accuracy 0.5921  macro_f1 0.4327  log_loss 0.9186  brier 0.5407  roc_auc 0.7022
test: accuracy 0.5092  macro_f1 0.3803  log_loss 1.0176  brier 0.6106  roc_auc 0.6255
test confusion matrix:
```
        pred H  pred D  pred A
true H     259       0      58
true D     127       0      70
true A     118       0     128
```
LR coefficients (standardised features; EloDiff/ProbHome collinear in `combined` - don't interpret those two):
```
                            H      D      A
AwayElo                 0.046 -0.061  0.015
EloDiff                 0.542 -0.059 -0.483
Form5Diff              -0.032 -0.034  0.066
Home_Avg5_ShotsFor      0.018 -0.035  0.018
Home_Avg5_ShotsAgainst -0.078 -0.019  0.096
Home_Avg5_TargetFor     0.049  0.053 -0.102
Away_Avg5_ShotsFor     -0.056 -0.037  0.093
Away_Avg5_ShotsAgainst  0.068 -0.017 -0.051
Home_H2H_GDPG          -0.020  0.032 -0.012
```

## LR-market  (10 candidates, 0s)
best params: {'C': 100}
val : accuracy 0.5974  macro_f1 0.4401  log_loss 0.9063  brier 0.5311  roc_auc 0.7208
test: accuracy 0.5171  macro_f1 0.3865  log_loss 1.0000  brier 0.5983  roc_auc 0.6460
test confusion matrix:
```
        pred H  pred D  pred A
true H     262       0      55
true D     124       0      73
true A     115       0     131
```
LR coefficients (standardised features; EloDiff/ProbHome collinear in `combined` - don't interpret those two):
```
               H      D      A
ProbHome   0.571  0.056 -0.627
ProbDraw  -0.034  0.145 -0.111
MaxOver25  0.011 -0.017  0.006
```

## LR-combined  (10 candidates, 0s)
best params: {'C': 100}
val : accuracy 0.6000  macro_f1 0.4405  log_loss 0.9048  brier 0.5301  roc_auc 0.7181
test: accuracy 0.5092  macro_f1 0.3808  log_loss 1.0045  brier 0.6010  roc_auc 0.6416
test confusion matrix:
```
        pred H  pred D  pred A
true H     257       0      60
true D     124       0      73
true A     116       0     130
```
LR coefficients (standardised features; EloDiff/ProbHome collinear in `combined` - don't interpret those two):
```
                            H      D      A
AwayElo                 0.058 -0.015 -0.043
EloDiff                 0.096 -0.029 -0.067
Form5Diff              -0.051 -0.033  0.084
Home_Avg5_ShotsFor     -0.015 -0.017  0.032
Home_Avg5_ShotsAgainst -0.043 -0.019  0.062
Home_Avg5_TargetFor     0.026  0.038 -0.064
Away_Avg5_ShotsFor     -0.028 -0.032  0.059
Away_Avg5_ShotsAgainst  0.034 -0.022 -0.012
Home_H2H_GDPG          -0.044  0.032  0.012
ProbHome                0.545  0.053 -0.597
ProbDraw               -0.023  0.148 -0.126
MaxOver25               0.014 -0.029  0.015
```

## RF-non_market  (120 candidates, 63s)
best params: {'max_depth': 10, 'min_samples_leaf': 30, 'max_features': 'sqrt', 'class_weight': None}
val : accuracy 0.5763  macro_f1 0.4194  log_loss 0.9272  brier 0.5458  roc_auc 0.7022
test: accuracy 0.5026  macro_f1 0.3740  log_loss 1.0197  brier 0.6122  roc_auc 0.6232
test confusion matrix:
```
        pred H  pred D  pred A
true H     260       0      57
true D     128       0      69
true A     123       1     122
```
RF impurity importances:
```
EloDiff                   0.408
AwayElo                   0.151
Home_Avg5_ShotsAgainst    0.076
Home_H2H_GDPG             0.076
Home_Avg5_ShotsFor        0.069
Away_Avg5_ShotsFor        0.065
Home_Avg5_TargetFor       0.054
Away_Avg5_ShotsAgainst    0.052
Form5Diff                 0.049
```

## RF-market  (80 candidates, 36s)
best params: {'max_depth': 3, 'min_samples_leaf': 1, 'max_features': 1.0, 'class_weight': None}
val : accuracy 0.5763  macro_f1 0.4243  log_loss 0.9114  brier 0.5345  roc_auc 0.7179
test: accuracy 0.5039  macro_f1 0.3761  log_loss 1.0065  brier 0.6034  roc_auc 0.6461
test confusion matrix:
```
        pred H  pred D  pred A
true H     256       0      61
true D     123       0      74
true A     119       0     127
```
RF impurity importances:
```
ProbHome     0.845
ProbDraw     0.144
MaxOver25    0.010
```

## RF-combined  (120 candidates, 80s)
best params: {'max_depth': 5, 'min_samples_leaf': 1, 'max_features': 0.5, 'class_weight': None}
val : accuracy 0.5711  macro_f1 0.4208  log_loss 0.9118  brier 0.5351  roc_auc 0.7158
test: accuracy 0.5079  macro_f1 0.3811  log_loss 1.0038  brier 0.6020  roc_auc 0.6470
test confusion matrix:
```
        pred H  pred D  pred A
true H     251       0      66
true D     119       0      78
true A     111       0     135
```
RF impurity importances:
```
ProbHome                  0.464
EloDiff                   0.207
ProbDraw                  0.117
AwayElo                   0.058
MaxOver25                 0.025
Home_Avg5_ShotsAgainst    0.025
Home_H2H_GDPG             0.024
Away_Avg5_ShotsFor        0.020
Home_Avg5_ShotsFor        0.019
Home_Avg5_TargetFor       0.015
Away_Avg5_ShotsAgainst    0.013
Form5Diff                 0.013
```

## Predicted champions (most expected points over the season)
Actual: 2024_25 Liverpool, 2025_26 Arsenal
```
model feature_set champion_2024_25 champion_2025_26
   LR  non_market        Liverpool          Arsenal
   LR      market        Liverpool          Arsenal
   LR    combined        Liverpool          Arsenal
   RF  non_market         Man City          Arsenal
   RF      market        Liverpool         Man City
   RF    combined        Liverpool          Arsenal
```

## Test summary
```
model feature_set  owner  accuracy  macro_f1  log_loss  brier  roc_auc champion_2024_25 champion_2025_26
   LR  non_market Dickon    0.5092    0.3803    1.0176 0.6106   0.6255        Liverpool          Arsenal
   LR      market Dickon    0.5171    0.3865    1.0000 0.5983   0.6460        Liverpool          Arsenal
   LR    combined Dickon    0.5092    0.3808    1.0045 0.6010   0.6416        Liverpool          Arsenal
   RF  non_market Dickon    0.5026    0.3740    1.0197 0.6122   0.6232         Man City          Arsenal
   RF      market Dickon    0.5039    0.3761    1.0065 0.6034   0.6461        Liverpool         Man City
   RF    combined Dickon    0.5079    0.3811    1.0038 0.6020   0.6470        Liverpool          Arsenal
```