# Step 3 - KNN + baselines (Faris)

Rows: train 6840, val 380, test 760
Classes order: ['H', 'D', 'A']. Selection metric: val log loss. Test evaluated once, no refit.

## AlwaysHome (baseline, no training)
probabilities = train class frequencies for every match: H 0.460, D 0.243, A 0.297
val : accuracy 0.4605  macro_f1 0.2102  log_loss 1.0558  brier 0.6380  roc_auc 0.5000
test: accuracy 0.4171  macro_f1 0.1962  log_loss 1.0835  brier 0.6568  roc_auc 0.5000
test confusion matrix:
```
        pred H  pred D  pred A
true H     317       0       0
true D     197       0       0
true A     246       0       0
```

## Bookmaker (baseline, no training)
probabilities = ['ProbHome', 'ProbDraw', 'ProbAway'] (Bet365, margin removed)
val : accuracy 0.5895  macro_f1 0.4346  log_loss 0.9093  brier 0.5331  roc_auc 0.7183
test: accuracy 0.5158  macro_f1 0.3863  log_loss 0.9945  brier 0.5952  roc_auc 0.6481
test confusion matrix:
```
        pred H  pred D  pred A
true H     258       0      59
true D     121       0      76
true A     112       0     134
```

## KNN-non_market  (24 candidates, 1s)
best params: {'n_neighbors': 200, 'weights': 'uniform'}
val : accuracy 0.5632  macro_f1 0.4092  log_loss 0.9341  brier 0.5509  roc_auc 0.7113
test: accuracy 0.4842  macro_f1 0.3622  log_loss 1.0280  brier 0.6168  roc_auc 0.6140
test confusion matrix:
```
        pred H  pred D  pred A
true H     243       0      74
true D     126       0      71
true A     121       0     125
```

## KNN-market  (24 candidates, 0s)
best params: {'n_neighbors': 100, 'weights': 'uniform'}
val : accuracy 0.5842  macro_f1 0.4575  log_loss 0.9044  brier 0.5281  roc_auc 0.7210
test: accuracy 0.5039  macro_f1 0.3934  log_loss 1.0063  brier 0.6023  roc_auc 0.6530
test confusion matrix:
```
        pred H  pred D  pred A
true H     252      10      55
true D     117       5      75
true A     110      10     126
```

## KNN-combined  (24 candidates, 1s)
best params: {'n_neighbors': 75, 'weights': 'uniform'}
val : accuracy 0.5921  macro_f1 0.4565  log_loss 0.9156  brier 0.5392  roc_auc 0.7151
test: accuracy 0.5013  macro_f1 0.3881  log_loss 1.0191  brier 0.6105  roc_auc 0.6344
test confusion matrix:
```
        pred H  pred D  pred A
true H     248       6      63
true D     119       4      74
true A     113       4     129
```

## Predicted champions (most expected points over the season)
Actual: 2024_25 Liverpool, 2025_26 Arsenal
```
     model feature_set champion_2024_25 champion_2025_26
AlwaysHome        none                -                -
 Bookmaker      bet365        Liverpool          Arsenal
       KNN  non_market        Liverpool          Arsenal
       KNN      market        Liverpool         Man City
       KNN    combined         Man City          Arsenal
```

## Test summary - all owners (results/test_metrics.csv)
```
     model feature_set  owner  accuracy  macro_f1  log_loss  brier  roc_auc champion_2024_25 champion_2025_26
        LR  non_market Dickon    0.5092    0.3803    1.0176 0.6106   0.6255        Liverpool          Arsenal
        LR      market Dickon    0.5171    0.3865    1.0000 0.5983   0.6460        Liverpool          Arsenal
        LR    combined Dickon    0.5092    0.3808    1.0045 0.6010   0.6416        Liverpool          Arsenal
        RF  non_market Dickon    0.5026    0.3740    1.0197 0.6122   0.6232         Man City          Arsenal
        RF      market Dickon    0.5039    0.3761    1.0065 0.6034   0.6461        Liverpool         Man City
        RF    combined Dickon    0.5079    0.3811    1.0038 0.6020   0.6470        Liverpool          Arsenal
AlwaysHome        none  Faris    0.4171    0.1962    1.0835 0.6568   0.5000                -                -
 Bookmaker      bet365  Faris    0.5158    0.3863    0.9945 0.5952   0.6481        Liverpool          Arsenal
       KNN  non_market  Faris    0.4842    0.3622    1.0280 0.6168   0.6140        Liverpool          Arsenal
       KNN      market  Faris    0.5039    0.3934    1.0063 0.6023   0.6530        Liverpool         Man City
       KNN    combined  Faris    0.5013    0.3881    1.0191 0.6105   0.6344         Man City          Arsenal
```