# Step 2 run log (auto-generated)

```
Loaded (7980, 67); train rows = 6840 (seasons 2005-2022)
Candidates: 44 non-market (A) + 17 market (M) = 61

## 2.0 Imputation
       column split  cells  fill_value                                                                                         reason
Home_H2H_GDPG train    179      0.0000 teams never met before: no H2H evidence -> neutral goal difference 0 (Home_H2H_n = 0 flags it)
Home_H2H_GDPG   val     10      0.0000 teams never met before: no H2H evidence -> neutral goal difference 0 (Home_H2H_n = 0 flags it)
 Home_H2H_PPG train    179      1.3481                     teams never met before: fill with train mean PPG (Home_H2H_n = 0 flags it)
 Home_H2H_PPG   val     10      1.3481                     teams never met before: fill with train mean PPG (Home_H2H_n = 0 flags it)
    HandiSize  test      1     -0.3000                                   handicap odds missing for a handful of matches: train median
    HandiSize train     10     -0.3000                                   handicap odds missing for a handful of matches: train median
    HandiHome train     10      1.9200                                   handicap odds missing for a handful of matches: train median
    HandiAway train     10      1.9500                                   handicap odds missing for a handful of matches: train median

## Relevance scores (train)

## 2.1 Redundancy: Spearman |rho| >= 0.85, clustered within each group
- cluster ['Home_H2H_PPG', 'Home_H2H_GDPG'] -> keep Home_H2H_GDPG
- cluster ['OddHome', 'ProbHome', 'ProbAway', 'OddAway', 'MaxHome', 'MaxAway', 'HandiSize'] -> keep ProbHome
- cluster ['OddDraw', 'ProbDraw', 'MaxDraw'] -> keep ProbDraw
- cluster ['Over25', 'Under25', 'MaxUnder25', 'MaxOver25'] -> keep MaxOver25
- cluster ['HandiHome', 'HandiAway'] -> keep HandiHome
61 -> 48 features after redundancy step

Strongest A-vs-M correlations (reported, not used to drop - we need A on its own):
EloDiff  MaxHome     0.944
         ProbHome    0.944
         MaxAway     0.944
         OddAway     0.943
         OddHome     0.943
         ProbAway    0.943

## 2.2 Feature vs class (Kruskal-Wallis, Bonferroni alpha=0.05; MI vs shuffled-target 95th pct)
                                   group    eps2  p_bonf      MI  MI_null95
ProbHome                      M (market)  0.1847  0.0000  0.1155     0.0069
EloDiff                   A (non-market)  0.1651  0.0000  0.1031     0.0063
HomeElo                   A (non-market)  0.0772  0.0000  0.0448     0.0069
AwayElo                   A (non-market)  0.0771  0.0000  0.0383     0.0123
Home_H2H_GDPG             A (non-market)  0.0766  0.0000  0.0484     0.0105
Form5Diff                 A (non-market)  0.0609  0.0000  0.0350     0.0102
Away_Avg5_ShotsFor        A (non-market)  0.0444  0.0000  0.0171     0.0073
Form3Diff                 A (non-market)  0.0435  0.0000  0.0257     0.0091
Home_Avg5_ShotsAgainst    A (non-market)  0.0386  0.0000  0.0196     0.0133
Away_Avg5_ShotsAgainst    A (non-market)  0.0349  0.0000  0.0169     0.0133
Home_Avg5_ShotsFor        A (non-market)  0.0325  0.0000  0.0174     0.0152
Home_Avg5_TargetFor       A (non-market)  0.0313  0.0000  0.0195     0.0083
Form5Home                 A (non-market)  0.0312  0.0000  0.0179     0.0116
ProbDraw                      M (market)  0.0292  0.0000  0.0309     0.0102
Form5Away                 A (non-market)  0.0292  0.0000  0.0174     0.0091
Away_Avg5_CornersAgainst  A (non-market)  0.0279  0.0000  0.0103     0.0158
Away_Avg5_GoalsFor        A (non-market)  0.0271  0.0000  0.0148     0.0115
Home_Avg5_GoalsFor        A (non-market)  0.0253  0.0000  0.0155     0.0084
Home_Avg5_CornersAgainst  A (non-market)  0.0247  0.0000  0.0166     0.0099
Away_Avg5_TargetAgainst   A (non-market)  0.0231  0.0000  0.0095     0.0100
Away_Avg5_TargetFor       A (non-market)  0.0219  0.0000  0.0111     0.0100
Away_Avg5_CornersFor      A (non-market)  0.0215  0.0000  0.0109     0.0035
Form3Home                 A (non-market)  0.0214  0.0000  0.0184     0.0126
Home_Avg5_CornersFor      A (non-market)  0.0214  0.0000  0.0135     0.0098
Form3Away                 A (non-market)  0.0210  0.0000  0.0130     0.0121
Home_Avg5_GoalsAgainst    A (non-market)  0.0201  0.0000  0.0101     0.0077
Away_Avg5_GoalsAgainst    A (non-market)  0.0175  0.0000  0.0040     0.0125
Home_Avg5_TargetAgainst   A (non-market)  0.0155  0.0000  0.0063     0.0071
MaxOver25                     M (market)  0.0138  0.0000  0.0182     0.0076
Away_Avg5_FoulsFor        A (non-market)  0.0061  0.0000  0.0060     0.0098
Home_Avg5_YellowAgainst   A (non-market)  0.0060  0.0000  0.0031     0.0081
HandiHome                     M (market)  0.0026  0.0029  0.0049     0.0089
Home_Avg5_FoulsFor        A (non-market)  0.0025  0.0047  0.0020     0.0090
Away_Avg5_YellowAgainst   A (non-market)  0.0021  0.0148  0.0040     0.0118
Away_Avg5_YellowFor       A (non-market)  0.0019  0.0333  0.0061     0.0069
Away_Avg5_RedFor          A (non-market)  0.0012  0.4053  0.0015     0.0065
EloChange2Home            A (non-market)  0.0010  0.6260  0.0089     0.0101
EloChange1Home            A (non-market)  0.0010  0.6375  0.0079     0.0060
Home_Avg5_RedAgainst      A (non-market)  0.0010  0.6501  0.0034     0.0091
Home_Avg5_YellowFor       A (non-market)  0.0007  1.0000  0.0053     0.0083
EloChange2Away            A (non-market)  0.0007  1.0000  0.0000     0.0065
BookmakerMargin               M (market)  0.0007  1.0000  0.0238     0.0091
Home_Avg5_RedFor          A (non-market)  0.0003  1.0000  0.0053     0.0061
Home_Avg5_FoulsAgainst    A (non-market)  0.0003  1.0000  0.0034     0.0078
Home_H2H_n                A (non-market)  0.0001  1.0000  0.0079     0.0064
Away_Avg5_FoulsAgainst    A (non-market)  0.0001  1.0000  0.0080     0.0063
EloChange1Away            A (non-market)  0.0000  1.0000  0.0035     0.0064
Away_Avg5_RedAgainst      A (non-market)  0.0000  1.0000  0.0033     0.0120
48 -> 39 features after relevance step

## 2.3 Boruta (RandomForest, class_weight=balanced, max_depth=7)

Boruta on A (non-market) (34 features, seeds (42, 7, 2026)): Confirmed 10, Tentative 0, Rejected 24; not unanimous across seeds: ['Home_Avg5_CornersAgainst', 'Away_Avg5_CornersAgainst']

Boruta on A+M (non-market + market) (39 features, seeds (42, 7, 2026)): Confirmed 13, Tentative 0, Rejected 26; not unanimous across seeds: ['Home_Avg5_TargetFor']

                              A run A conf/seeds    A+M run A+M conf/seeds
AwayElo                   Confirmed          3/3  Confirmed            3/3
Away_Avg5_CornersAgainst   Rejected          1/3   Rejected            0/3
Away_Avg5_CornersFor       Rejected          0/3   Rejected            0/3
Away_Avg5_FoulsAgainst     Rejected          0/3   Rejected            0/3
Away_Avg5_FoulsFor         Rejected          0/3   Rejected            0/3
Away_Avg5_GoalsAgainst     Rejected          0/3   Rejected            0/3
Away_Avg5_GoalsFor         Rejected          0/3   Rejected            0/3
Away_Avg5_ShotsAgainst    Confirmed          3/3  Confirmed            3/3
Away_Avg5_ShotsFor        Confirmed          3/3  Confirmed            3/3
Away_Avg5_TargetAgainst    Rejected          0/3   Rejected            0/3
Away_Avg5_TargetFor        Rejected          0/3   Rejected            0/3
Away_Avg5_YellowAgainst    Rejected          0/3   Rejected            0/3
Away_Avg5_YellowFor        Rejected          0/3   Rejected            0/3
BookmakerMargin                   -            -   Rejected            0/3
EloChange1Home             Rejected          0/3   Rejected            0/3
EloDiff                   Confirmed          3/3  Confirmed            3/3
Form3Away                  Rejected          0/3   Rejected            0/3
Form3Diff                  Rejected          0/3   Rejected            0/3
Form3Home                  Rejected          0/3   Rejected            0/3
Form5Away                  Rejected          0/3   Rejected            0/3
Form5Diff                 Confirmed          3/3  Confirmed            3/3
Form5Home                  Rejected          0/3   Rejected            0/3
HandiHome                         -            -   Rejected            0/3
HomeElo                   Confirmed          3/3  Confirmed            3/3
Home_Avg5_CornersAgainst   Rejected          1/3   Rejected            0/3
Home_Avg5_CornersFor       Rejected          0/3   Rejected            0/3
Home_Avg5_FoulsFor         Rejected          0/3   Rejected            0/3
Home_Avg5_GoalsAgainst     Rejected          0/3   Rejected            0/3
Home_Avg5_GoalsFor         Rejected          0/3   Rejected            0/3
Home_Avg5_ShotsAgainst    Confirmed          3/3  Confirmed            3/3
Home_Avg5_ShotsFor        Confirmed          3/3  Confirmed            3/3
Home_Avg5_TargetAgainst    Rejected          0/3   Rejected            0/3
Home_Avg5_TargetFor       Confirmed          3/3  Confirmed            2/3
Home_Avg5_YellowAgainst    Rejected          0/3   Rejected            0/3
Home_H2H_GDPG             Confirmed          3/3  Confirmed            3/3
Home_H2H_n                 Rejected          0/3   Rejected            0/3
MaxOver25                         -            -  Confirmed            3/3
ProbDraw                          -            -  Confirmed            3/3
ProbHome                          -            -  Confirmed            3/3

## 2.4 Multicollinearity (VIF)

A: VIF before pruning:
HomeElo                   inf
AwayElo                   inf
EloDiff                   inf
Home_Avg5_ShotsFor        2.2
Home_H2H_GDPG             1.9
Home_Avg5_TargetFor       1.8
Form5Diff                 1.7
Home_Avg5_ShotsAgainst    1.6
Away_Avg5_ShotsFor        1.6
Away_Avg5_ShotsAgainst    1.6
- drop HomeElo (VIF inf; lowest Boruta importance among VIF>=10: ['HomeElo', 'AwayElo', 'EloDiff'])
A: max VIF after pruning = 4.8

AM: VIF before pruning:
HomeElo                    inf
AwayElo                    inf
EloDiff                    inf
ProbHome                  13.5
ProbDraw                   3.4
MaxOver25                  3.0
Home_Avg5_ShotsFor         2.3
Home_H2H_GDPG              2.0
Form5Diff                  1.8
Home_Avg5_TargetFor        1.8
Home_Avg5_ShotsAgainst     1.7
Away_Avg5_ShotsFor         1.7
Away_Avg5_ShotsAgainst     1.7
- drop HomeElo (VIF inf; lowest Boruta importance among VIF>=10: ['HomeElo', 'AwayElo', 'EloDiff', 'ProbHome'])
- drop EloDiff (VIF 12.2; lowest Boruta importance among VIF>=10: ['EloDiff', 'ProbHome'])
AM: max VIF after pruning = 6.1

FINAL A  (9): ['AwayElo', 'EloDiff', 'Form5Diff', 'Home_Avg5_ShotsFor', 'Home_Avg5_ShotsAgainst', 'Home_Avg5_TargetFor', 'Away_Avg5_ShotsFor', 'Away_Avg5_ShotsAgainst', 'Home_H2H_GDPG']
FINAL AM (11): ['AwayElo', 'Form5Diff', 'Home_Avg5_ShotsFor', 'Home_Avg5_ShotsAgainst', 'Home_Avg5_TargetFor', 'Away_Avg5_ShotsFor', 'Away_Avg5_ShotsAgainst', 'Home_H2H_GDPG', 'ProbHome', 'ProbDraw', 'MaxOver25']
Tentative: []
```
